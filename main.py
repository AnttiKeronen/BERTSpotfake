import torch
import pandas as pd
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
import os

from data_loader import FakeNewsDataset


from models import TextOnlyModel
from train_val import train

root_dir = r"C:\Users\keron\OneDrive\Työpöytä\SpotFakefull\twitter"

df_train = pd.read_csv(os.path.join(root_dir, "train_posts_clean.csv"))
df_test = pd.read_csv(os.path.join(root_dir, "test_posts.csv"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

MAX_LEN = 500

train_dataset = FakeNewsDataset(df_train, root_dir, None, tokenizer, MAX_LEN)
val_dataset = FakeNewsDataset(df_test, root_dir, None, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

loss_fn = nn.BCELoss()

model = TextOnlyModel().to(device)

optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-8)
total_steps = len(train_loader) * 5

scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

train(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=5,
    evaluation=True,
    device=device,
    param_dict_model=None,
    param_dict_opt=None,
    save_best=True,
    file_path=os.path.join(root_dir, "saved_models", "bert_only.pt"),
    writer=None
)
