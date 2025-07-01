Multi-label Document Classification Microservice

A Dockerized microservice to classify and auto-tag scanned documents using a fine-tuned `distilBERT` model. The service provides a REST API and supports model evaluation, retraining, and inference through a CLI.

---

## 🚀 Features

- Fine-tunes `distilBERT` for multi-label classification using HuggingFace Transformers.
- Preprocessing pipeline supports `.json` and `.txt` formats.
- Exposes a Flask-based REST API for predictions.
- CLI support for inference and evaluation.
- Lightweight retraining with cronjob support.
- Dockerized and CI/CD-ready.

---

## 🗂️ Project Structure

```bash
DocumentTagger/
├── app/
│   ├── data_preprocessing.py     # Loads and tokenizes dataset
│   ├── inference.py              # CLI entrypoint for prediction
│   ├── evaluate.py               # Evaluation metrics (F1, report)
│   ├── model.py                  # Model load/save logic
│   ├── config.py                 # Paths and configs
├── Dockerfile
├── Makefile
├── requirements.txt
├── processed.json                # Sample dataset
└── README.md
