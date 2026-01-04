import torch
import pandas as pd
import sys
import os

print("="*60)
print("DIAGNOSTICS")
print("="*60)
print(f"Python: {sys.version.split()[0]}")
print(f"Torch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
else:
    print("WARNING: CUDA is NOT available.")

print("\n" + "="*60)
print("CHECKING DATA LOADING")
print("="*60)

csv_path = "data/IMDB Dataset.csv"
if not os.path.exists(csv_path):
    print(f"ERROR: File not found at {os.path.abspath(csv_path)}")
else:
    print(f"File found at: {csv_path}")
    try:
        print("Attempting to read first 5 rows with pandas...")
        df = pd.read_csv(csv_path, nrows=5)
        print("SUCCESS! Data loaded correctly.")
        print("-" * 20)
        print(df[['review', 'sentiment']])
        print("-" * 20)
    except Exception as e:
        print(f"FAILED to read CSV: {e}")
        print("Traceback:")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("Diagnostics Complete")
print("="*60)
