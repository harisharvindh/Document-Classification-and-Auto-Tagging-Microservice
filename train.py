# train.py

import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import load_dataset, Dataset
import pandas as pd
import json
import os

# Load and preprocess data
def load_data(json_path):
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    texts = [entry["text"] for entry in raw_data]
    labels = [entry["labels"] for entry in raw_data]

    mlb = MultiLabelBinarizer()
    binary_labels = mlb.fit_transform(labels)

    return Dataset.from_dict({
        "text": texts,
        "labels": binary_labels.tolist()
    }), mlb.classes_

# Tokenize
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

# Configuration
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 8
EPOCHS = 3
json_train_path = "data/train.json"

# Load dataset and tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
train_dataset_raw, label_names = load_data(json_train_path)
train_dataset = train_dataset_raw.map(tokenize, batched=True)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Define model
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_names),
    problem_type="multi_label_classification"
)

# Training setup
args = TrainingArguments(
    output_dir="outputs",
    evaluation_strategy="no",
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    save_total_limit=1,
    logging_dir="logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
)

# Train and save
trainer.train()
model.save_pretrained("model")
tokenizer.save_pretrained("model")

with open("model/label_names.txt", "w") as f:
    f.write("\n".join(label_names))
