import sys
import os
import re

import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def preprocess_text(text):
    import string
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = " ".join(text.split())
    return text

MODEL_PATH = "svm_model.joblib"
VECTORIZER_PATH = "tfidf_vectorizer.joblib"

def find_model():
    paths = [
        (MODEL_PATH, VECTORIZER_PATH),
        ("models/" + MODEL_PATH, "models/" + VECTORIZER_PATH),
    ]
    for m_path, v_path in paths:
        if os.path.exists(m_path) and os.path.exists(v_path):
            return joblib.load(m_path), joblib.load(v_path)
    return None, None

def check_legitimacy(text):
    text_lower = text.lower()
    
    strong_legit = [
        r'\b(?:reuters|associated press|ap)\b',
        r'\(\s*reuters\s*\)|\(\s*ap\s*\)',
        r'\b(?:washington|london|beijing|moscow)\s*\(',
    ]
    
    legitimacy_indicators = [
        r'\b(?:said|stated|according to|reported|told reporters)\b',
        r'\b(?:spokesperson|official|minister|prime minister|president|ceo)\b',
        r'\b(?:on monday|on tuesday|on wednesday|on thursday|on friday|on saturday|on sunday)\b',
        r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}',
        r'\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)',
        r'\b(?:announced|confirmed|declared)\b',
        r'\b(?:2024|2025)\b',
        r'\b(?:gaza|ukraine|israel|palestine)\b',
        r'\b(?:covid|pandemic|vaccine)\b',
        r'\b(?:election|vote|campaign|candidate)\b',
    ]
    
    fake_indicators = [
        r'\b(?:breaking|urgent|shocking|unbelievable)\b',
        r'!{2,}',
        r'\b(?:image|video shows)\b',
    ]
    
    strong_legit_score = sum(1 for pattern in strong_legit if re.search(pattern, text_lower))
    legit_score = sum(1 for pattern in legitimacy_indicators if re.search(pattern, text_lower))
    fake_score = sum(1 for pattern in fake_indicators if re.search(pattern, text_lower))
    
    return strong_legit_score, legit_score, fake_score

def predict(text, model, vectorizer):
    sample_clean = [preprocess_text(text)]
    sample_vec = vectorizer.transform(sample_clean)
    
    pred = model.predict(sample_vec)[0]
    proba = model.predict_proba(sample_vec)[0]
    
    strong_legit, legit_score, fake_score = check_legitimacy(text)
    
    proba_fake = proba[1]
    proba_real = proba[0]
    
   
    if strong_legit >= 1:
        is_fake = False
        confidence = 0.75
    elif legit_score >= 2:
        is_fake = False
        confidence = max(0.6, proba_real)
    elif proba_fake >= 0.75 and fake_score >= 1:
        is_fake = True
        confidence = proba_fake
    elif proba_fake >= 0.85:
        is_fake = True
        confidence = proba_fake
    else:

        is_fake = False
        confidence = proba_real
    
    score = 100 - confidence * 100 if is_fake else confidence * 100
    
    return {
        'label': "Fake" if is_fake else "Real",
        'is_fake': is_fake,
        'score': score,
        'confidence': confidence,
        'proba_fake': proba_fake,
        'proba_real': proba_real,
        'strong_legit': strong_legit,
        'legit_score': legit_score,
        'fake_score': fake_score
    }

def main():
    st.set_page_config(page_title="News Credibility Checker")
    
    st.title("Intelligent News Credibility Analysis Model")
    st.markdown("Analyze news articles using SVM Linear Model")
    
    model, vectorizer = find_model()
    
    if not model:
        st.error("Models not found")
        st.info("Run training first")
        return
    
    text = st.text_area("Enter news text:", height=200)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        analyze = st.button("Analyze", type="primary")
    with col2:
        clear = st.button("Clear")
    
    if clear:
        st.session_state.text_input = ""
        st.rerun()
    
    if analyze:
        if not text.strip():
            st.warning("Enter text")
        elif len(text.strip()) < 30:
            st.warning("Enter at least 30 characters")
        else:
            with st.spinner("Analyzing..."):
                result = predict(text, model, vectorizer)
            
            st.markdown("---")
            
            if result['is_fake']:
                st.error(f"{result['label']} News")
            else:
                st.success(f"{result['label']} News")
            
            st.metric("Credibility Score", f"{result['score']:.0f}/100")
            st.progress(result['score'] / 100)
            st.metric("Confidence", f"{result['confidence']:.1%}")
            
            with st.expander("Debug Info"):
                st.write(f"Legitimacy indicators: {result['legit_score']}")
                st.write(f"Fake indicators: {result['fake_score']}")


main()
