import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import re
import os


# ----------- TEXT CLEANING ------------
def text_preprocessing(text):
    text = re.sub(r'(@.*?)[\s]', ' ', text)     # remove @someone
    text = re.sub(r'&amp;', '&', text)         # fix &amp;
    text = re.sub(r'\s+', ' ', text).strip()   # remove extra spaces
    return text


# ----------- DATASET CLASS ------------
class FakeNewsDataset(Dataset):

    def __init__(self, df, root_dir, image_transform, tokenizer, MAX_LEN):
        """
        IMPORTANT:
        - image_transform is ignored (BERT only)
        - root_dir is ignored (no images used)
        """

        self.df = df
        self.tokenizer = tokenizer
        self.max_len = MAX_LEN

    def __len__(self):
        return len(self.df)

    def encode_text(self, text):
        encoded = self.tokenizer.encode_plus(
            text_preprocessing(text),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
        )
        input_ids = torch.tensor(encoded["input_ids"])
        mask = torch.tensor(encoded["attention_mask"])
        return input_ids, mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        input_ids, att_mask = self.encode_text(row["post_text"])

        label = 1 if row["label"] == "fake" else 0
        label = torch.tensor(label, dtype=torch.float)

        return {
            "input_ids": input_ids,
            "attention_mask": att_mask,
            "label": label
        }
