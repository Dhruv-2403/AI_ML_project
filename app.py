import streamlit as st
import joblib
import re
import string
import numpy as np


st.set_page_config(page_title="Fake News Detector", layout="centered")


@st.cache_resource
def load_model():
    model = joblib.load("svm_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    return model, vectorizer


svm_model, tfidf_vec = load_model()


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = " ".join(text.split())
    return text


