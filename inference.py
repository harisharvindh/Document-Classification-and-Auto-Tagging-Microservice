# inference.py

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import json

def load_model(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def predict(text, model, tokenizer, label_binarizer_path, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).detach().cpu().numpy()[0]

    with open(label_binarizer_path, 'r') as f:
        classes = json.load(f)["classes"]

    predictions = [label for label, prob in zip(classes, probs) if prob > threshold]
    return predictions, probs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Text input for inference")
    parser.add_argument("--model", type=str, default="model", help="Path to fine-tuned model")
    parser.add_argument("--labels", type=str, default="label_binarizer.json", help="Path to class labels (JSON)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold")

    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    predictions, probs = predict(args.text, model, tokenizer, args.labels, args.threshold)

    print("\nPredicted Tags:", predictions)
    print("Probabilities:", np.round(probs, 3))
