import argparse
import json
import torch
from sklearn.metrics import classification_report, f1_score
from transformers import AutoTokenizer
from model import load_model
from data_preprocessing import encode_labels, multilabel_binarizer
from config import MODEL_NAME

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, tokenizer, dataset_path):
    with open(dataset_path, "r") as f:
        data = json.load(f)

    texts = [item["text"] for item in data]
    true_labels = [item["labels"] for item in data]
    binarized_true = multilabel_binarizer.fit_transform(true_labels)

    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt").to(device)
    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(**encodings)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)

    print("=== Classification Report ===")
    print(classification_report(binarized_true, preds, target_names=multilabel_binarizer.classes_))
    print("Micro F1 Score:", f1_score(binarized_true, preds, average='micro'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned document classification model.")
    parser.add_argument("--data", type=str, required=True, help="Path to the test dataset JSON file")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model()
    evaluate_model(model, tokenizer, args.data)
