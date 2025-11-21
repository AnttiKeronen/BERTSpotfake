import os
import torch
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from data_loader import FakeNewsDataset
from models import TextOnlyModel
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# ===== PATHS =====
root_dir = r"C:\Users\lauri\Desktop\BERTSpotfake"
twitter_dir = os.path.join(root_dir, "twitter")
test_csv = os.path.join(twitter_dir, "test_posts.csv")
saved_model_path = os.path.join(root_dir, "saved_models", "bert_only.pt")

# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ===== LOAD FULL TEST SET =====
df_test = pd.read_csv(test_csv)
print(f"Using FULL test set: {len(df_test)} samples")

# ===== TOKENIZER =====
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
MAX_LEN = 500

# ===== DATASET (TEXT ONLY) =====
val_dataset = FakeNewsDataset(
    df_test,
    root_dir=twitter_dir,      # ignored
    image_transform=None,      # ignored
    tokenizer=tokenizer,
    MAX_LEN=MAX_LEN
)

val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# ===== LOAD MODEL =====
model = TextOnlyModel().to(device)
checkpoint = torch.load(saved_model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded! Starting evaluation...")

loss_fn = torch.nn.BCELoss()
all_labels, all_preds, all_probs = [], [], []
val_loss = 0

# ===== TEST LOOP =====
with torch.no_grad():
    for batch in val_loader:

        text_ids = batch["input_ids"].to(device)
        text_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(text_ids, text_mask)
        outputs = outputs.float()

        loss = loss_fn(outputs, labels)
        val_loss += loss.item()

        preds = (outputs >= 0.5).float()

        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())
        all_probs.append(outputs.cpu())


# Combine all results
all_labels = torch.cat(all_labels).numpy()
all_preds = torch.cat(all_preds).numpy()
all_probs = torch.cat(all_probs).numpy()

# ===== METRICS =====
accuracy = (all_labels == all_preds).mean() * 100
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)
val_loss /= len(val_loader)

print(f"TEST LOSS: {val_loss:.4f}")
print(f"TEST ACCURACY: {accuracy:.2f}%")
print(f"TEST PRECISION: {precision:.4f}")
print(f"TEST RECALL: {recall:.4f}")
print(f"TEST F1 SCORE: {f1:.4f}")
print(f"TEST AUC: {auc:.4f}")

# ===== ROC CURVE =====
fpr, tpr, _ = roc_curve(all_labels, all_probs)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (BERT Text Only)")
plt.legend(loc="lower right")
plt.show()


