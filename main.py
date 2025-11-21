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

# ======= PATHS =======
repo_root = r"C:\Users\lauri\Desktop\BERTSpotfake"
twitter_dir = os.path.join(repo_root, "twitter")

df_train = pd.read_csv(os.path.join(twitter_dir, "train_posts_clean.csv"))
df_test = pd.read_csv(os.path.join(twitter_dir, "test_posts.csv"))

# ======= DEVICE =======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======= TOKENIZER =======
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
MAX_LEN = 500

# ======= DATASETS (BERT-only → no images) =======
train_dataset = FakeNewsDataset(
    df_train,
    None,            # img_path
    None,            # transform
    tokenizer,
    MAX_LEN
)

val_dataset = FakeNewsDataset(
    df_test,
    None,
    None,
    tokenizer,
    MAX_LEN
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

# ======= LOSS =======
loss_fn = nn.BCELoss()

# ======= MODEL =======
model = TextOnlyModel(fine_tune_text=True).to(device)


# ======= OPTIMIZER & SCHEDULER =======
optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-8)
total_steps = len(train_loader) * 5

scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

# ======= SAVE PATH =======
save_folder = os.path.join(repo_root, "saved_models")
os.makedirs(save_folder, exist_ok=True)

save_path = os.path.join(save_folder, "bert_only.pt")

# ======= TRAIN =======
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
    file_path=save_path,
    writer=None
)
