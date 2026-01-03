from train_dpo import train_dpo
from train_ppo import train_ppo
from eval_utils import evaluate_toxicity, save_evaluation_plots
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_NAME, DEVICE, EVAL_MODEL
import argparse


def main(run_dpo=True, run_ppo=True):
    """
    Main entry point for DPO and PPO training comparison.
    
    Args:
        run_dpo: Whether to run DPO training
        run_ppo: Whether to run PPO training
    """
    dpo_model = None
    dpo_tokenizer = None
    dpo_logs = []
    ppo_logs = []
    
    # 1. Train DPO
    if run_dpo:
        print("\n" + "="*60)
        print("Starting DPO Training...")
        print("="*60)
        dpo_model, dpo_tokenizer, dpo_logs = train_dpo()
    
    # 2. Train PPO
    if run_ppo:
        print("\n" + "="*60)
        print("Starting PPO Training...")
        print("="*60)
        _, _, ppo_logs = train_ppo()
    
    # 3. Load reference model for win rate computation
    print("\nLoading reference model for evaluation...")
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    ref_model.eval()
    
    # Load tokenizer if not already loaded
    if dpo_tokenizer is None:
        dpo_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        dpo_tokenizer.pad_token = dpo_tokenizer.eos_token
    
    # Load DPO model if not already loaded (in case we only ran PPO)
    if dpo_model is None and run_dpo:
        print("Loading trained DPO model...")
        dpo_model = AutoModelForCausalLM.from_pretrained("./dpo_toxic_reduced_model").to(DEVICE)
    
    # 4. Generate Plots with real data
    print("\n" + "="*60)
    print("Generating evaluation plots...")
    print("="*60)
    
    if dpo_logs:
        save_evaluation_plots(
            dpo_logs=dpo_logs,
            ppo_logs=ppo_logs if ppo_logs else None,
            dpo_model=dpo_model,
            ref_model=ref_model,
            tokenizer=dpo_tokenizer,
            device=DEVICE
        )
    
    # 5. Final Evaluation
    if dpo_model is not None:
        print("\n" + "="*60)
        print("Final DPO Model Evaluation")
        print("="*60)
        evaluate_toxicity(dpo_model, dpo_tokenizer, DEVICE, EVAL_MODEL)
    
    print("\n" + "="*60)
    print("All training and evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DPO and/or PPO models")
    parser.add_argument("--dpo-only", action="store_true", help="Run only DPO training")
    parser.add_argument("--ppo-only", action="store_true", help="Run only PPO training")
    args = parser.parse_args()
    
    if args.dpo_only:
        main(run_dpo=True, run_ppo=False)
    elif args.ppo_only:
        main(run_dpo=False, run_ppo=True)
    else:
        main(run_dpo=True, run_ppo=True)
