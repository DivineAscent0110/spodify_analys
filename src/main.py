import torch
from transformers import AutoTokenizer
import pandas as pd
from src.config import Config
from src.preprocess import clean_data
from src.model import TransformerClassifier
from src.trainer import train
from torch.utils.data import DataLoader, Dataset

class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['processed_review_text']
        label = self.df.iloc[idx]['review_rating'] - 1  # Assuming 1-5 ratings
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {key: val.squeeze() for key, val in encoding.items()}, label

def main():
    cfg = Config()
    df = pd.read_csv('data/review_data.csv')
    df = clean_data(df)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    dataset = ReviewDataset(df, tokenizer, cfg.max_seq_length)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = TransformerClassifier(cfg.model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train(model, dataloader, cfg, device)

if __name__ == "__main__":
    main()

