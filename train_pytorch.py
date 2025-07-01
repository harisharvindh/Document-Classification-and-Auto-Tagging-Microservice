# train_pytorch.py

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from typing import List
from config import Config
from utils import load_data, set_seed
from labels import LABELS

set_seed(Config.seed)

class DocumentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[List[int]], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze() for key, val in encoded.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = {k: v.to(device) for k, v in batch.items()}
            labels = inputs.pop('labels')
            logits = model(**inputs).logits
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds) > 0.5
    all_labels = np.vstack(all_labels)
    return f1_score(all_labels, all_preds, average='micro')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.model_name,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        id2label={str(i): label for i, label in enumerate(LABELS)},
        label2id={label: i for i, label in enumerate(LABELS)}
    ).to(device)

    train_texts, train_labels = load_data(Config.train_file, LABELS)
    val_texts, val_labels = load_data(Config.val_file, LABELS)

    train_dataset = DocumentDataset(train_texts, train_labels, tokenizer)
    val_dataset = DocumentDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=Config.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.eval_batch_size)

    optimizer = AdamW(model.parameters(), lr=Config.learning_rate)

    for epoch in range(Config.num_train_epochs):
        print(f"\nEpoch {epoch+1}/{Config.num_train_epochs}")
        train_loss = train(model, train_loader, optimizer, device)
        val_f1 = evaluate(model, val_loader, device)
        print(f"Train loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")

    os.makedirs(Config.output_dir, exist_ok=True)
    model.save_pretrained(Config.output_dir)
    tokenizer.save_pretrained(Config.output_dir)
    print(f"\nModel saved to {Config.output_dir}")

if __name__ == "__main__":
    main()
