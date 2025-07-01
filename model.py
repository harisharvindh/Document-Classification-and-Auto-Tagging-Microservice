# model.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from typing import List, Dict
import logging
import os
import time

try:
    from config import LABELS  # Optional: For validation or metadata
except ImportError:
    LABELS = None

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

class TaggingModel:
    def __init__(self, model_name: str = "distilbert-base-uncased", model_path: str = None, threshold: float = 0.5, device: str = None):
        """
        Initializes the tokenizer and model. Supports loading from HuggingFace or local directory.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading model from {'local path' if model_path else 'HuggingFace Hub'}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path or model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path or model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e

        self.id2label = self.model.config.id2label if hasattr(self.model.config, 'id2label') else None
        if not self.id2label:
            raise ValueError("Model config does not contain id2label mapping.")

        if LABELS and len(LABELS) != len(self.id2label):
            logger.warning("Mismatch between LABELS in config.py and model config id2label!")

    def predict_single(self, text: str) -> Dict[str, float]:
        """
        Predicts multi-label tags for a single document.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        predictions = {
            self.id2label[i]: float(prob)
            for i, prob in enumerate(probs)
            if prob >= self.threshold
        }

        if not predictions:
            predictions = {"None": 1.0}  # fallback if nothing crosses threshold

        return predictions

    def predict_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Predicts tags for a batch of documents.
        """
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.sigmoid(logits).cpu().numpy()

        batch_predictions = []
        for prob_row in probs:
            labels = {
                self.id2label[i]: float(p)
                for i, p in enumerate(prob_row)
                if p >= self.threshold
            }
            if not labels:
                labels = {"None": 1.0}
            batch_predictions.append(labels)

        return batch_predictions

    def get_model_info(self) -> Dict[str, str]:
        """
        Returns basic model metadata.
        """
        return {
            "model_name": self.model.config._name_or_path,
            "num_labels": str(self.model.config.num_labels),
            "threshold": str(self.threshold),
            "device": self.device,
        }


if __name__ == "__main__":
    # Example usage
    model = TaggingModel(model_path="your-local-or-hf-model-path", threshold=0.4)
    sample_texts = [
        "The tenant agrees to vacate the premises within 30 days.",
        "This report summarizes financial performance in Q2."
    ]
    predictions = model.predict_batch(sample_texts)
    for i, pred in enumerate(predictions):
        print(f"Doc {i+1}: {pred}")
