import torch

# --- CONFIGURATION ---
MODEL_NAME = "distilgpt2"
EVAL_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# DPO Hyperparameters
BETA = 0.1          # The KL-divergence penalty coefficient
LR = 5e-5           # Learning rate
BATCH_SIZE = 4
MAX_LENGTH = 64     # Max sequence length for training
CSV_PATH = "data/IMDB Dataset.csv" # Update this path for local use

# PPO Hyperparameters
PPO_CLIP_EPS = 0.2      # Clipping epsilon for PPO objective
PPO_EPOCHS = 4          # Number of PPO epochs per batch
KL_COEF = 0.1           # KL penalty coefficient
ENTROPY_COEF = 0.01     # Entropy bonus coefficient

