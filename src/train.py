import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.data_loader import load_data
from utils.feature_extraction import extract_features
from config.settings import Config

def train_model():
    print("Loading data...")
   
    df = load_data(Config.DATASET_PATH)
    df = df.sample(20000, random_state=42)  
   
    print(f"Loaded {len(df)} articles")
    
    print("Extracting features...")
    X, vectorizer = extract_features(df["clean"])
    y = df["label"]
    print(f"Feature matrix shape: {X.shape}")
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    print("Training model...")
    lr = LogisticRegression(max_iter=1500, C=1.5, solver="saga", random_state=42, n_jobs=-1)
    lr.fit(X_train, y_train)
    print("Training complete!")
    
    print("Evaluating...")
    y_pred = lr.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    
    print("Saving model...")
    joblib.dump(lr, "../models/lr_model.joblib")
    joblib.dump(vectorizer, "../models/tfidf_vectorizer.joblib")
    print("Model saved!")

if __name__ == "__main__":
    train_model()
