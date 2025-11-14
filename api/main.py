from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from pathlib import Path

from Script.train_imdb_ann import SimpleTextANN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("models/ann_imdb_final.pt")
TOKENIZER_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
vocab_size = len(tokenizer.get_vocab())
pad_id = tokenizer.pad_token_id or 0

EMBED_DIM = 200
HIDDEN_DIM = 256
DROPOUT = 0.4

model = SimpleTextANN(vocab_size=vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, pad_id=pad_id, dropout=DROPOUT)
state = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state)
model.to(DEVICE)
model.eval()

app = FastAPI(title="IMDB ANN Sentiment API")

class Payload(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {"service": "ann-imdb", "version": "0.1.0", "python": f"{'.'.join(map(str, _import_('sys').version_info[:3]))}"}

@app.post("/predict")
def predict(payload: Payload):
    txt = payload.text
    enc = tokenizer(txt, truncation=True, padding=True, max_length=200, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        prob = torch.sigmoid(logits).cpu().numpy().tolist()[0]
        label = int(prob >= 0.5)

    return {"input_text": txt, "label": label, "probability": float(prob)}