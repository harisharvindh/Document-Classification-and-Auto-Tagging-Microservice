# data_preprocessing.py

import pandas as pd
import json
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer

def load_dataset(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df

def preprocess_data(df, text_col='text', labels_col='labels', tokenizer_name='distilbert-base-uncased', max_len=512):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df[labels_col])
    
    encodings = tokenizer(
        df[text_col].tolist(),
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt"
    )
    
    return encodings, y, mlb

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to dataset JSON file")
    args = parser.parse_args()
    
    df = load_dataset(args.input)
    encodings, y, mlb = preprocess_data(df)
    print("Dataset loaded and tokenized. Shape:", len(y))
