from torch.utils.data import Dataset
import pandas as pd
import random
from config import MAX_LENGTH

class IMDbToxicReductionDataset(Dataset):
    """
    Transforms the IMDb dataset into DPO preference pairs.
    SCENARIO A LOGIC:
    - Winner (y_w) = Positive Review (Safe/Constructive)
    - Loser  (y_l) = Negative Review (Toxic/Hateful)
    """
    def __init__(self, csv_path, tokenizer, max_samples=1000):
        self.tokenizer = tokenizer
        self.data = []

        print(f"Loading data from {csv_path}...")
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print("Error: 'IMDB Dataset.csv' not found. Please provide the correct path.")
            # Depending on use case, might want to raise error or return empty
            return

        # Separate into "Safe" (Positive) and "Toxic" (Negative) pools
        safe_reviews = df[df['sentiment'] == 'positive']['review'].tolist()
        toxic_reviews = df[df['sentiment'] == 'negative']['review'].tolist()

        # Shuffle to ensure random pairing
        random.shuffle(safe_reviews)
        random.shuffle(toxic_reviews)

        # Create Pairs
        count = min(len(safe_reviews), len(toxic_reviews), max_samples)

        print("Constructing Preference Pairs (Winner=Safe, Loser=Toxic)...")
        for i in range(count):
            safe_text = safe_reviews[i]
            toxic_text = toxic_reviews[i]

            # Tokenize
            tokenized_w = tokenizer(
                safe_text, truncation=True, max_length=MAX_LENGTH, padding='max_length', return_tensors='pt'
            )
            tokenized_l = tokenizer(
                toxic_text, truncation=True, max_length=MAX_LENGTH, padding='max_length', return_tensors='pt'
            )

            self.data.append({
                'winner_input_ids': tokenized_w['input_ids'].squeeze(0),
                'winner_attention_mask': tokenized_w['attention_mask'].squeeze(0),
                'loser_input_ids': tokenized_l['input_ids'].squeeze(0),
                'loser_attention_mask': tokenized_l['attention_mask'].squeeze(0),
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
