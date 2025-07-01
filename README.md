# Document Auto-Tagging Microservice

This project implements a production-ready microservice to classify and auto-tag scanned documents (contracts, reports, etc.) using a fine-tuned `distilBERT` model. It supports multi-label classification, reproducible training workflows, batch inference, and containerized deployment.

## Features

- Fine-tuned `distilBERT` for multi-label document tagging
- FastAPI-based inference API
- CLI-based batch/single document inference
- Preprocessing from raw JSON to model-ready format
- Training with both HuggingFace Trainer and custom PyTorch loop
- MLOps workflow using `Docker`, `Makefile`, and reproducible training
- Lightweight evaluation and retraining routines

---

## 📁 Directory Structure



## 🗂️ Project Structure

```bash
├── app.py                # FastAPI app for serving the model
├── config.py             # Configuration variables (paths, model, thresholds)
├── data_preprocessing.py # Preprocesses raw JSON into model-friendly format
├── Dockerfile            # Docker setup for containerizing the service
├── evaluate.py           # Evaluate model predictions on test set
├── inference.py          # CLI script for batch/single prediction
├── labels.txt            # Label list used by the model
├── Makefile              # Task automation (train, clean, docker, etc.)
├── model.py              # Core class wrapping tokenizer/model/prediction
├── processed.json        # Preprocessed dataset (sample)
├── README.md             # Project documentation
├── requirements.txt      # Python package dependencies
├── train.py              # General-purpose training script
├── train_hf.py           # HuggingFace Trainer-based training
├── train_pytorch.py      # Manual PyTorch training loop



---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt

make train_hf

make train_pytorch

python inference.py --text "This agreement outlines the tenant’s responsibilities..."

uvicorn app:app --reload

make docker_build
make docker_run




