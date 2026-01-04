import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import random

from config import (
    MODEL_NAME, EVAL_MODEL, DEVICE, BETA, LR,
    BATCH_SIZE, MAX_LENGTH, CSV_PATH, MAX_SAMPLES, EVAL_STEPS
)
from dataset import IMDbToxicReductionDataset
from dpo_utils import dpo_loss, get_batch_log_probs
from eval_utils import evaluate_policy, compute_kl_divergence, TEST_PROMPTS

def train_dpo(
    csv_path=CSV_PATH,
    eval_model_name=EVAL_MODEL,
    test_prompts=TEST_PROMPTS,
    eval_steps=EVAL_STEPS
):
    print("\n--- INITIALIZING DPO TRAINING WITH EVALUATION ---")

    # A. Load Tokenizer & Models
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Policy Model
    policy_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    policy_model.train()

    # Reference Model (Frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Safety Judge (Evaluation Oracle)
    safety_judge = pipeline(
        "sentiment-analysis",
        model=eval_model_name,
        device=0 if DEVICE == "cuda" else -1
    )

    # B. Prepare Data
    dataset = IMDbToxicReductionDataset(csv_path, tokenizer, max_samples=MAX_SAMPLES)
    if len(dataset) == 0:
        print("Dataset is empty. Aborting training.")
        return policy_model, tokenizer, []

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=LR)

    # Evaluation Storage
    eval_logs = []

    # C. Training Loop
    epochs = 1
    global_step = 0

    print(f"Starting training on {len(dataset)} pairs for {epochs} epoch(s)...")

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for batch in progress_bar:
            optimizer.zero_grad()

            # Move batch to GPU
            w_ids = batch['winner_input_ids'].to(DEVICE)
            w_mask = batch['winner_attention_mask'].to(DEVICE)
            l_ids = batch['loser_input_ids'].to(DEVICE)
            l_mask = batch['loser_attention_mask'].to(DEVICE)

            # --- 1. Policy Log Probs ---
            policy_w_logps = get_batch_log_probs(policy_model, w_ids, w_mask)
            policy_l_logps = get_batch_log_probs(policy_model, l_ids, l_mask)
            policy_logprobs = torch.stack([policy_w_logps, policy_l_logps], dim=1)

            # --- 2. Reference Log Probs ---
            with torch.no_grad():
                ref_w_logps = get_batch_log_probs(ref_model, w_ids, w_mask)
                ref_l_logps = get_batch_log_probs(ref_model, l_ids, l_mask)
                ref_logprobs = torch.stack([ref_w_logps, ref_l_logps], dim=1)

            # --- 3. DPO Loss ---
            loss = dpo_loss(policy_logprobs, ref_logprobs, beta=BETA)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

            # --- 4. Evaluation (No Gradient) ---
            if global_step % eval_steps == 0:
                with torch.no_grad():
                    policy_model.eval()
                    
                    # Use first 20 fixed prompts for consistent evaluation
                    EVAL_PROMPTS = test_prompts[:20] if len(test_prompts) >= 20 else test_prompts

                    # Compute KL using unified method (same as PPO)
                    kl_estimate = compute_kl_divergence(
                        policy_model, ref_model, tokenizer, DEVICE,
                        EVAL_PROMPTS, num_samples=10
                    )

                    safety_metrics = evaluate_policy(
                        policy_model, tokenizer, safety_judge, EVAL_PROMPTS, DEVICE
                    )

                    eval_logs.append({
                        "step": global_step,
                        "loss": loss.item(),
                        "kl": kl_estimate,
                        **safety_metrics
                    })
                    
                    policy_model.train()

            global_step += 1

    # D. Save Model
    print("Training Complete. Saving Model...")
    policy_model.save_pretrained("./models/dpo_toxic_reduced_model")
    tokenizer.save_pretrained("./models/dpo_toxic_reduced_model")

    return policy_model, tokenizer, eval_logs
