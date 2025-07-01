# config.py

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA = os.path.join(DATA_DIR, "processed.json")
LABELS_FILE = os.path.join(DATA_DIR, "labels.txt")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_NAME = "distilbert-base-uncased"

# Training Parameters
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5

# Label Information
LABELS = [
    "contract", "consulting", "IT", "finance", "report", "quarterly", "meeting",
    "minutes", "annual", "invoice", "software", "development", "memo", "HR", "policy",
    "patent", "blockchain", "technology", "legal", "compliance", "survey",
    "feedback", "customer", "board", "merger", "training", "onboarding", "security"
]
NUM_LABELS = len(LABELS)

# Create MODEL_DIR if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)
