# train_hf.py

import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import Dataset
from config import Config
from typing import List


def load_data(json_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)
    texts = [item['text'] for item in data]
    labels = [item['labels'] for item in data]
    return texts, labels


def encode_labels(label_list: List[List[str]], all_labels: List[str]):
    mlb = MultiLabelBinarizer(classes=all_labels)
    encoded = mlb.fit_transform(label_list)
    return encoded, mlb


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)


if __name__ == "__main__":
    cfg = Config()

    # Load data
    texts, raw_labels = load_data(cfg.data_path)
    label_names = cfg.label_names
    encoded_labels, mlb = encode_labels(raw_labels, label_names)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=len(label_names), problem_type="multi_label_classification")

    # Prepare dataset
    dataset = Dataset.from_dict({"text": texts, "labels": encoded_labels.tolist()})
    dataset = dataset.map(tokenize_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True
    )

    # Custom compute metrics (optional)
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = torch.sigmoid(torch.tensor(logits)).numpy()
        preds = (preds >= 0.5).astype(int)
        acc = (preds == labels).mean()
        return {"accuracy": acc}

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()
    model.save_pretrained(cfg.save_path)
    tokenizer.save_pretrained(cfg.save_path)
