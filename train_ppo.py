import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import random

from config import (
    MODEL_NAME, EVAL_MODEL, DEVICE, LR,
    MAX_LENGTH, CSV_PATH,
    PPO_CLIP_EPS, PPO_EPOCHS, KL_COEF, ENTROPY_COEF
)
from ppo_utils import (
    generate_and_score,
    compute_log_probs_for_generated,
    ppo_loss,
    compute_kl_divergence,
    compute_entropy
)
from eval_utils import evaluate_policy, TEST_PROMPTS


# Training prompts for generation
TRAINING_PROMPTS = [
    "The movie was", "I thought the plot", "The director really",
    "Honestly, this film", "The actors were", "The story seemed",
    "In my opinion", "The ending was", "What I liked was",
    "The cinematography", "The soundtrack was", "Overall I felt",
    "The main character", "The dialogue felt", "The pacing was"
]


def train_ppo(
    csv_path=CSV_PATH,
    eval_model_name=EVAL_MODEL,
    test_prompts=TEST_PROMPTS,
    eval_steps=10,
    num_episodes=30,
    batch_size=8
):
    """
    Train a language model using Proximal Policy Optimization (PPO).
    
    Unlike DPO which uses preference pairs, PPO:
    1. Generates completions from the current policy
    2. Scores them with a reward model (sentiment classifier)
    3. Updates the policy using clipped policy gradients
    """
    print("\n--- INITIALIZING PPO TRAINING ---")
    
    # A. Load Tokenizer & Models
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Policy Model (the one we're training)
    policy_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    policy_model.train()
    
    # Reference Model (frozen, for KL penalty)
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # Reward Model (Sentiment Classifier - acts as our reward signal)
    reward_judge = pipeline(
        "sentiment-analysis",
        model=eval_model_name,
        device=0 if DEVICE == "cuda" else -1
    )
    
    # Safety Judge for evaluation (same model, different purpose)
    safety_judge = pipeline(
        "sentiment-analysis", 
        model=eval_model_name,
        device=0 if DEVICE == "cuda" else -1
    )
    
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=LR)
    
    # Evaluation Storage
    eval_logs = []
    
    # B. Training Loop
    print(f"Starting PPO training for {num_episodes} episodes...")
    
    progress_bar = tqdm(range(num_episodes), desc="PPO Training")
    
    for episode in progress_bar:
        # Sample prompts for this episode
        episode_prompts = random.sample(TRAINING_PROMPTS, min(batch_size, len(TRAINING_PROMPTS)))
        
        # Generate completions and get rewards
        full_ids, full_masks, rewards, generated_texts = generate_and_score(
            policy_model, tokenizer, episode_prompts, reward_judge, DEVICE
        )
        
        # Compute old log probs (before update)
        with torch.no_grad():
            old_policy_logprobs = compute_log_probs_for_generated(
                policy_model, full_ids, full_masks, None
            )
            ref_logprobs = compute_log_probs_for_generated(
                ref_model, full_ids, full_masks, None
            )
        
        # PPO update epochs
        for ppo_epoch in range(PPO_EPOCHS):
            optimizer.zero_grad()
            
            # Current policy log probs
            policy_logprobs = compute_log_probs_for_generated(
                policy_model, full_ids, full_masks, None
            )
            
            # Compute probability ratio
            ratio = torch.exp(policy_logprobs - old_policy_logprobs)
            
            # Advantages = rewards - baseline (we use rewards directly, centered)
            advantages = rewards - rewards.mean()
            if advantages.std() > 0:
                advantages = advantages / (advantages.std() + 1e-8)
            
            # PPO clipped loss
            policy_loss = ppo_loss(advantages, ratio, clip_eps=PPO_CLIP_EPS)
            
            # KL penalty (keep close to reference)
            kl_div = compute_kl_divergence(policy_logprobs, ref_logprobs)
            kl_penalty = KL_COEF * kl_div
            
            # Entropy bonus (encourage exploration)
            outputs = policy_model(input_ids=full_ids, attention_mask=full_masks)
            entropy = compute_entropy(outputs.logits, full_masks)
            entropy_bonus = -ENTROPY_COEF * entropy
            
            # Total loss
            total_loss = policy_loss + kl_penalty + entropy_bonus
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
        
        progress_bar.set_postfix({
            'loss': total_loss.item(),
            'reward': rewards.mean().item(),
            'kl': kl_div.item()
        })
        
        # Evaluation
        if episode % eval_steps == 0:
            with torch.no_grad():
                # Use the same 20 fixed prompts as DPO for fair comparison
                EVAL_PROMPTS = test_prompts[:20] if len(test_prompts) >= 20 else test_prompts
                
                safety_metrics = evaluate_policy(
                    policy_model,
                    tokenizer,
                    safety_judge,
                    EVAL_PROMPTS,
                    DEVICE
                )
                
                eval_logs.append({
                    "step": episode,
                    "loss": total_loss.item(),
                    "kl": kl_div.item(),
                    "reward": rewards.mean().item(),
                    **safety_metrics
                })
    
    # C. Save Model
    print("PPO Training Complete. Saving Model...")
    policy_model.save_pretrained("./models/ppo_toxic_reduced_model")
    tokenizer.save_pretrained("./models/ppo_toxic_reduced_model")
    
    return policy_model, tokenizer, eval_logs


if __name__ == "__main__":
    model, tokenizer, logs = train_ppo()
    print(f"\nTraining complete! Logged {len(logs)} evaluation points.")
