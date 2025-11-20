import torch
import pandas as pd
import numpy as np
import transformers
import torchvision
from torchvision import models, transforms
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

import random
import time
import os
import re
import math




def train(
    model,
    loss_fn,
    optimizer,
    scheduler,
    train_dataloader,
    val_dataloader,
    epochs,
    evaluation,
    device,
    param_dict_model,
    param_dict_opt,
    save_best,
    file_path,
    writer
):

    print("Start training...\n")
    print(" Epoch  |  Batch  |  Train Loss  |  Val Loss  |  Val Acc  |  Elapsed")
    print("----------------------------------------------------------------------------------------------------")

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()

        total_loss = 0

        for step, batch in enumerate(train_dataloader):

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device).float()

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)

            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 20 == 0 and step != 0:
                elapsed = time.time() - start_time
                print(f"epoch {epoch:3} | batch {step:4} | loss {loss.item():.6f} |   -    |   -    | elapsed {elapsed:.2f}")

        # ---------- VALIDATION ----------
        if evaluation:
            val_loss, val_acc = evaluate(model, val_dataloader, loss_fn, device)

            print(f"epoch {epoch:3} |   end   | train_loss {total_loss/len(train_dataloader):.6f} | val_loss {val_loss:.6f} | val_acc {val_acc:.4f}")

            if val_loss < best_val_loss and save_best:
                best_val_loss = val_loss
                torch.save({"model_state_dict": model.state_dict()}, file_path)
                print("✔ Saved new best model!")



def evaluate(model, dataloader, loss_fn, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device).float()

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            preds = (outputs >= 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy
