import pandas as pd
from utils.preprocessing import preprocess_text

def load_data(path):
    df = pd.read_csv(path)
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["content"] = df["title"] + " " + df["text"]
    df = df[df["content"].str.strip().astype(bool)].reset_index(drop=True)
    df["clean"] = df["content"].apply(preprocess_text)
    return df
