import pandas as pd
import torch

from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config.config import MODEL_NAME, MAX_SEQ_LEN


class LogDataset(Dataset):

    def __init__(self, csv_file):

        self.data = pd.read_csv(str(csv_file))

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        encodings = self.tokenizer(
            self.data["Text"].astype(str).tolist(),
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LEN,
            return_tensors="pt"
        )

        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = torch.tensor(
            self.data["Label"].values,
            dtype=torch.long
        )

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):

        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }