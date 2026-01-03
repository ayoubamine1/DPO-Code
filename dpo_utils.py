import torch
import torch.nn.functional as F

def dpo_loss(policy_logprobs, ref_logprobs, beta):
    """
    The Direct Preference Optimization Loss.
    Equation: -log sigmoid( beta * (log(pi_theta/pi_ref)_w - log(pi_theta/pi_ref)_l) )
    """
    log_ratio_winner = policy_logprobs[:, 0] - ref_logprobs[:, 0]
    log_ratio_loser  = policy_logprobs[:, 1] - ref_logprobs[:, 1]

    logits = beta * (log_ratio_winner - log_ratio_loser)

    loss = -F.logsigmoid(logits).mean()
    return loss

def get_batch_log_probs(model, input_ids, attention_mask):
    """
    Computes log probabilities for the generated tokens.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits # Shape: (Batch, Seq_Len, Vocab)

    # Shift logits: logit at index t predicts token at index t+1
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    # Compute log_softmax to get log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather only the log probability of the actual target token
    # gather dim=2 means we select the vocab index corresponding to shift_labels
    token_log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Mask out padding tokens so they don't affect the sum
    shift_mask = attention_mask[..., 1:].contiguous()

    # Sum over the sequence length to get log p(y|x)
    sequence_log_probs = (token_log_probs * shift_mask).sum(dim=-1)

    return sequence_log_probs
