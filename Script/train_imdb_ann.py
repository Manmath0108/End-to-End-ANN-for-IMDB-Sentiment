import os
import argparse
import random
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# optional libs
try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

try:
    from transformers import AutoTokenizer, DataCollatorWithPadding
except Exception:
    raise RuntimeError("Install transformers: pip install transformers")

class SimpleTextANN(nn.Module):
    def __init__(self, vocab_size, embed_dim=200, hidden_dim=256, pad_id=0, dropout=0.4):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, input_ids, attention_mask=None):
        emb = self.embedding(input_ids)  

        if attention_mask is None:
            mask = (input_ids != self.embedding.padding_idx).float()
        else:
            mask = attention_mask.float()

        mask = mask.unsqueeze(-1)         
        emb_masked = emb * mask        
        summed = emb_masked.sum(1)       
        denom = mask.sum(1).clamp(min=1e-9)  
        pooled = summed / denom            

        x = self.dropout(pooled)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x).squeeze(-1)   
        return logits

class IMDBHFDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=200):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        txt = str(self.texts[idx])
        encoded = self.tokenizer(
            txt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=True,
        )
        encoded["labels"] = int(self.labels[idx])
        return encoded

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_imdb_subset(max_per_label=1000, seed=42):
    if load_dataset is None:
        raise RuntimeError("Install datasets: pip install datasets")

    ds = load_dataset("imdb")

    def collect(split, k):
        texts, labels = [], []
        count = {0: 0, 1: 0}
        for ex in ds[split]:
            y = int(ex["label"])
            if count[y] < k:
                texts.append(ex["text"])
                labels.append(y)
                count[y] += 1
            if count[0] >= k and count[1] >= k:
                break
        return texts, labels

    train_texts, train_labels = collect("train", max_per_label)
    test_texts, test_labels = collect("test", max_per_label // 2)
    return train_texts, train_labels, test_texts, test_labels

def train_and_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)
    set_seed(args.seed)

    print("Loading IMDB subset ...")
    train_texts, train_labels, test_texts, test_labels = load_imdb_subset(
        max_per_label=args.max_train_per_label, seed=args.seed
    )
    print(f"Train size = {len(train_texts)}, Test size = {len(test_texts)}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    collator  = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", return_tensors="pt")

    train_ds = IMDBHFDataset(train_texts, train_labels, tokenizer, max_length=args.max_len)
    test_ds  = IMDBHFDataset(test_texts, test_labels, tokenizer, max_length=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    pad_id = tokenizer.pad_token_id or 0

    print("Tokenizer =", args.tokenizer_name)
    print("Vocab size =", vocab_size)

    model = SimpleTextANN(
        vocab_size,
        args.embed_dim,
        args.hidden_dim,
        pad_id,
        args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, verbose=True
    )

    best_val = float("inf")
    no_improve = 0
    os.makedirs(args.output_dir, exist_ok=True)
    best_path = Path(args.output_dir) / "ann_imdb_best.pt"

    print("\n=== TRAINING START ===")
    for epoch in range(1, args.epochs + 1):

        model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch in train_loader:
            optimizer.zero_grad()

            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].float().to(device)

            logits = model(ids, mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            train_correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

        train_loss /= total
        train_acc  = train_correct / total

        model.eval()
        val_loss = 0
        val_correct = 0
        v_total = 0

        with torch.no_grad():
            for batch in test_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].float().to(device)

                logits = model(ids, mask)
                loss = loss_fn(logits, labels)

                val_loss += loss.item() * labels.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).long()
                val_correct += (preds == labels.long()).sum().item()
                v_total += labels.size(0)

        val_loss /= v_total
        val_acc = val_correct / v_total

        print(f"Epoch {epoch:02d}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            print(f"> Saved best model -> {best_path}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print("Early stopping triggered.")
                break

    print("\n=== FINAL EVAL ===")
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        print("Loaded best model.")

    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy().tolist()

            logits = model(ids, mask)
            preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int).tolist()

            y_true.extend(labels)
            y_pred.extend(preds)

    try:
        from sklearn.metrics import classification_report, accuracy_score
        print("Final accuracy:", accuracy_score(y_true, y_pred))
        print(classification_report(y_true, y_pred, digits=4))
    except Exception:
        acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)
        print("Final accuracy:", acc)

    final_path = Path(args.output_dir) / "ann_imdb_final.pt"
    torch.save(model.state_dict(), final_path)
    print("Saved final model ->", final_path)

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=200)
    p.add_argument("--embed_dim", type=int, default=200)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tokenizer_name", type=str, default="distilbert-base-uncased")
    p.add_argument("--glove_path", type=str, default=None)
    p.add_argument("--freeze_emb", action="store_true")
    p.add_argument("--output_dir", type=str, default="models")
    p.add_argument("--max_train_per_label", type=int, default=1000)
    p.add_argument("--cpu", action="store_true")

    args = p.parse_args()
    pprint(vars(args))
    train_and_eval(args)

if __name__ == "__main__":
    main()