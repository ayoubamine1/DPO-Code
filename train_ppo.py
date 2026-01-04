"""
PPO Training for Toxicity Reduction - Manual Implementation.

This implements PPO with Actor-Critic architecture from scratch:
- Actor (Policy): GPT-2 model that generates completions
- Critic (Value Head): Simple linear layer for value estimation  
- Reward Model: BERT sentiment classifier as reward signal

This avoids TRL library issues by implementing the core PPO algorithm directly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

from config import MODEL_NAME, EVAL_MODEL, DEVICE, MAX_SAMPLES, EVAL_STEPS, PPO_LR, KL_COEF
from eval_utils import evaluate_policy, compute_kl_divergence, TEST_PROMPTS


class ValueHead(nn.Module):
    """Simple value head for PPO critic."""
    def __init__(self, hidden_size):
        super().__init__()
        self.summary = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, hidden_states):
        # Take the last hidden state
        return self.summary(hidden_states[:, -1, :]).squeeze(-1)


class ActorCritic(nn.Module):
    """Actor-Critic model combining GPT-2 with a value head."""
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.value_head = ValueHead(self.model.config.n_embd)
        
        # Freeze most of GPT-2, only train last layers + value head
        for param in self.model.parameters():
            param.requires_grad = False
        # Only unfreeze last transformer block
        for param in self.model.transformer.h[-1].parameters():
            param.requires_grad = True
        for param in self.model.transformer.ln_f.parameters():
            param.requires_grad = True
        for param in self.model.lm_head.parameters():
            param.requires_grad = True
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=True
        )
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]
        values = self.value_head(hidden_states)
        return logits, values
    
    def generate(self, **kwargs):
        return self.model.generate(**kwargs)


def get_log_probs(logits, labels, mask):
    """Compute log probabilities for the generated tokens."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = mask[..., 1:].contiguous()
    
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Sum over sequence (masked) to match DPO's "total sequence log-prob" scaling
    masked_log_probs = token_log_probs * shift_mask
    seq_log_probs = masked_log_probs.sum(dim=-1) # / (shift_mask.sum(dim=-1) + 1e-8)
    
    return seq_log_probs


def ppo_loss(old_log_probs, new_log_probs, advantages, clip_eps=0.2):
    """Compute clipped PPO policy loss with numerical stability."""
    # Clamp the log prob difference to prevent exploding gradients
    log_ratio = torch.clamp(new_log_probs - old_log_probs, -10.0, 10.0)
    ratio = torch.exp(log_ratio)
    
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    return policy_loss


def train_ppo(
    csv_path="data/IMDB Dataset.csv",
    max_samples=MAX_SAMPLES,  # Same as DPO
    num_epochs=1,      # Same as DPO
    batch_size=4,      # Same as DPO
    eval_steps=EVAL_STEPS,     # Same as DPO for aligned step counts
    lr=PPO_LR,           # Use config LR
    clip_eps=0.2,
    kl_coef=KL_COEF,      # Use config KL coef
    value_coef=0.1     # Lower value coefficient
):
    """
    Train a language model using PPO (manual implementation).
    
    Unlike DPO which uses preference pairs, PPO:
    1. Generates completions from the current policy (Actor)
    2. Scores them with a reward model (BERT classifier)
    3. Updates policy using clipped PPO with value baseline (Critic)
    """
    print("\n--- INITIALIZING PPO TRAINING (MANUAL) ---")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Load Actor-Critic Model
    print(f"Loading model: {MODEL_NAME}")
    actor_critic = ActorCritic(MODEL_NAME).to(DEVICE)
    
    # Reference model (frozen) for KL penalty
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # 3. Load Reward Model (BERT Sentiment Classifier)
    print(f"Loading reward model: {EVAL_MODEL}")
    reward_pipe = pipeline(
        "sentiment-analysis",
        model=EVAL_MODEL,
        device=0 if DEVICE == "cuda" else -1
    )
    
    safety_judge = pipeline(
        "sentiment-analysis",
        model=EVAL_MODEL,
        device=0 if DEVICE == "cuda" else -1
    )
    
    # 4. Load prompts from dataset
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    prompts = [" ".join(text.split()[:6]) for text in df['review'].iloc[:max_samples].tolist()]
    print(f"Created {len(prompts)} prompts for PPO training")
    
    # 5. Optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, actor_critic.parameters()),
        lr=lr,
        weight_decay=0.01
    )
    
    # Evaluation logs
    eval_logs = []
    global_step = 0
    
    # 6. Training Loop
    print("Starting PPO Training...")
    
    for epoch in range(num_epochs):
        total_reward = 0
        total_loss = 0
        num_batches = 0
        
        # Shuffle prompts each epoch
        import random
        random.shuffle(prompts)
        
        progress_bar = tqdm(range(0, len(prompts), batch_size), desc=f"Epoch {epoch+1}")
        
        for i in progress_bar:
            batch_prompts = prompts[i:i+batch_size]
            if len(batch_prompts) < batch_size:
                continue  # Skip incomplete batches
            
            # A. Generate completions (Rollout) - with gradient disabled
            actor_critic.eval()
            generated_texts = []
            all_input_ids = []
            
            with torch.no_grad():
                for prompt in batch_prompts:
                    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
                    
                    output_ids = actor_critic.generate(
                        **inputs,
                        max_new_tokens=20,
                        do_sample=True,
                        top_k=50,
                        temperature=1.0,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    
                    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    generated_texts.append(text)
                    all_input_ids.append(output_ids[0])
            
            # Pad sequences
            max_len = max(ids.size(0) for ids in all_input_ids)
            padded_ids = []
            padded_masks = []
            for ids in all_input_ids:
                pad_len = max_len - ids.size(0)
                if pad_len > 0:
                    ids = F.pad(ids, (0, pad_len), value=tokenizer.eos_token_id)
                    mask = torch.cat([torch.ones(max_len - pad_len), torch.zeros(pad_len)])
                else:
                    mask = torch.ones(max_len)
                padded_ids.append(ids)
                padded_masks.append(mask)
            
            input_ids = torch.stack(padded_ids).to(DEVICE)
            attention_mask = torch.stack(padded_masks).to(DEVICE)
            
            # B. Compute Rewards (using BERT classifier)
            rewards = []
            for text in generated_texts:
                try:
                    result = reward_pipe(text)[0]
                    if result['label'] == 'POSITIVE':
                        reward = result['score']
                    else:
                        reward = 1.0 - result['score']
                except:
                    reward = 0.5  # Default neutral reward on error
                rewards.append(reward)
            
            rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
            total_reward += rewards.mean().item()
            
            # C. Compute old log probs and values (detached, before update)
            with torch.no_grad():
                old_logits, old_values = actor_critic(input_ids, attention_mask)
                old_log_probs = get_log_probs(old_logits, input_ids, attention_mask)
                
                # Reference model log probs for KL
                ref_outputs = ref_model(input_ids, attention_mask=attention_mask)
                ref_log_probs = get_log_probs(ref_outputs.logits, input_ids, attention_mask)
            
            # D. Compute advantages (simple: rewards - baseline value)
            # Use rewards centered around 0.5
            rewards_normalized = (rewards - 0.5) * 2  # Scale to [-1, 1]
            advantages = rewards_normalized - old_values.detach()
            
            # Normalize advantages for stability
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # E. PPO Update
            actor_critic.train()
            optimizer.zero_grad()
            
            # New forward pass
            new_logits, new_values = actor_critic(input_ids, attention_mask)
            new_log_probs = get_log_probs(new_logits, input_ids, attention_mask)
            
            # Policy loss (clipped PPO)
            policy_loss = ppo_loss(old_log_probs, new_log_probs, advantages, clip_eps)
            
            # Value loss (MSE between predicted values and normalized rewards)
            value_loss = F.mse_loss(new_values, rewards_normalized.detach())
            
            # KL penalty (encourage staying close to reference)
            kl_div = (old_log_probs - ref_log_probs).mean()
            
            # Total loss with clipping
            loss = policy_loss + value_coef * value_loss + kl_coef * kl_div
            
            # Check for NaN/Inf and skip if detected
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWarning: NaN/Inf loss detected at step {global_step}, skipping batch")
                continue
            
            loss.backward()
            
            # Aggressive gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 0.5)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'reward': f'{rewards.mean().item():.3f}'
            })
            
            # F. Evaluation (same frequency as DPO)
            if global_step % eval_steps == 0:
                with torch.no_grad():
                    actor_critic.eval()
                    EVAL_PROMPTS = TEST_PROMPTS[:20] if len(TEST_PROMPTS) >= 20 else TEST_PROMPTS
                    
                    # Compute KL using unified method (same as DPO)
                    kl_estimate = compute_kl_divergence(
                        actor_critic.model, ref_model, tokenizer, DEVICE,
                        EVAL_PROMPTS, num_samples=10
                    )

                    safety_metrics = evaluate_policy(
                        actor_critic.model, tokenizer, safety_judge, EVAL_PROMPTS, DEVICE
                    )
                    
                    eval_logs.append({
                        "step": global_step,
                        "loss": loss.item(),
                        "kl": kl_estimate,
                        "reward": rewards.mean().item(),
                        **safety_metrics
                    })
            
            global_step += 1
        
        if num_batches > 0:
            print(f"Epoch {epoch+1}: Avg Loss={total_loss/num_batches:.4f}, "
                  f"Avg Reward={total_reward/num_batches:.4f}")
    
    # 7. Save Model
    print("\nPPO Training Complete. Saving Model...")
    actor_critic.model.save_pretrained("./models/ppo_toxic_reduced_model")
    tokenizer.save_pretrained("./models/ppo_toxic_reduced_model")
    
    return actor_critic.model, tokenizer, eval_logs
