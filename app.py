# app.py

from flask import Flask, request, jsonify
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import os

app = Flask(__name__)

# Load model and tokenizer
MODEL_PATH = "model"
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)

# Load label names
with open(os.path.join(MODEL_PATH, "label_names.txt"), "r") as f:
    label_names = f.read().splitlines()

# Inference route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if "text" not in data:
        return jsonify({"error": "Missing 'text' field in JSON input"}), 400

    text = data["text"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).squeeze().tolist()

    threshold = data.get("threshold", 0.5)
    predicted_labels = [label_names[i] for i, p in enumerate(probs) if p >= threshold]

    return jsonify({
        "text": text,
        "predicted_labels": predicted_labels,
        "probabilities": {label_names[i]: float(p) for i, p in enumerate(probs)}
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
