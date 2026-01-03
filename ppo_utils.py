import torch
import torch.nn.functional as F
from transformers import pipeline
import numpy as np


def get_reward_from_judge(judge, texts):
    """
    Get rewards from the sentiment judge.
    Positive sentiment = positive reward, Negative = negative reward.
    
    Args:
        judge: HuggingFace sentiment-analysis pipeline
        texts: List of generated texts
    
    Returns:
        rewards: Tensor of reward values [-1, 1]
    """
    results = judge(texts)
    rewards = []
    for result in results:
        if result['label'] == 'POSITIVE':
            reward = result['score']  # Positive reward [0.5, 1]
        else:
            reward = -result['score']  # Negative reward [-1, -0.5]
        rewards.append(reward)
    return torch.tensor(rewards, dtype=torch.float32)


def compute_log_probs_for_generated(model, input_ids, attention_mask, generated_ids):
    """
    Compute log probabilities for tokens that were generated (not the prompt).
    
    Args:
        model: The language model
        input_ids: Full sequence (prompt + generated)
        attention_mask: Attention mask
        generated_ids: The generated token IDs (after the prompt)
    
    Returns:
        log_probs: Sum of log probabilities for the generated tokens
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Mask for attention
    shift_mask = attention_mask[..., 1:].contiguous()
    
    # Sum log probs for the generated portion only
    sequence_log_probs = (token_log_probs * shift_mask).sum(dim=-1)
    
    return sequence_log_probs


def ppo_loss(advantages, ratio, clip_eps=0.2):
    """
    Compute the PPO clipped surrogate objective.
    
    L^CLIP = E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]
    
    Args:
        advantages: Advantage estimates (batch_size,)
        ratio: pi_theta(a|s) / pi_old(a|s) (batch_size,)
        clip_eps: Clipping epsilon
    
    Returns:
        loss: Negative of the clipped objective (for minimization)
    """
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    
    # Take the minimum of clipped and unclipped
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    
    return policy_loss


def compute_kl_divergence(policy_logprobs, ref_logprobs):
    """
    Compute KL divergence between policy and reference.
    KL(π_θ || π_ref) ≈ avg(log π_θ - log π_ref)
    """
    return (policy_logprobs - ref_logprobs).mean()


def compute_entropy(logits, mask):
    """
    Compute entropy of the policy for exploration bonus.
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    
    # Apply mask and average
    shift_mask = mask[..., :-1]
    masked_entropy = (entropy[..., :-1] * shift_mask).sum(dim=-1) / shift_mask.sum(dim=-1).clamp(min=1)
    
    return masked_entropy.mean()


def generate_and_score(model, tokenizer, prompts, judge, device, max_new_tokens=20):
    """
    Generate completions and score them with the reward model.
    
    Returns:
        full_ids: Full sequence IDs (prompt + generated)
        full_masks: Attention masks
        rewards: Reward scores from judge
        generated_texts: List of generated text strings
    """
    model.eval()
    all_ids = []
    all_masks = []
    all_texts = []
    
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )
            
            generated_ids = outputs.sequences
            attention_mask = torch.ones_like(generated_ids)
            
            all_ids.append(generated_ids)
            all_masks.append(attention_mask)
            
            text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            all_texts.append(text)
    
    model.train()
    
    # Stack tensors
    max_len = max(ids.size(1) for ids in all_ids)
    padded_ids = []
    padded_masks = []
    
    for ids, mask in zip(all_ids, all_masks):
        pad_len = max_len - ids.size(1)
        if pad_len > 0:
            ids = F.pad(ids, (0, pad_len), value=tokenizer.eos_token_id)
            mask = F.pad(mask, (0, pad_len), value=0)
        padded_ids.append(ids)
        padded_masks.append(mask)
    
    full_ids = torch.cat(padded_ids, dim=0)
    full_masks = torch.cat(padded_masks, dim=0)
    
    # Get rewards from judge
    rewards = get_reward_from_judge(judge, all_texts)
    
    return full_ids, full_masks, rewards.to(device), all_texts
