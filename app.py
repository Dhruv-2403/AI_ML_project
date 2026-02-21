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


def get_top_keywords(clean_text, n=8):
    feature_names = tfidf_vec.get_feature_names_out()
    coefs = svm_model.calibrated_classifiers_[0].estimator.coef_[0]
    vec = tfidf_vec.transform([clean_text])
    nonzero = vec.nonzero()[1]
    scored = [(feature_names[i], coefs[i], vec[0, i]) for i in nonzero]
    scored.sort(key=lambda x: abs(x[1] * x[2]), reverse=True)
    return scored[:n]


st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        text-align: center;
        padding: 0.8rem 0 0.2rem 0;
        color: #c62828;
    }
    .subtitle {
        text-align: center;
        color: #28a745;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .result-real {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-left: 6px solid #28a745;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .result-fake {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border-left: 6px solid #dc3545;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .chip-real {
        display: inline-block;
        background: #e8f5e9;
        color: #2e7d32;
        padding: 4px 12px;
        border-radius: 16px;
        margin: 3px;
        font-size: 0.85rem;
    }
    .chip-fake {
        display: inline-block;
        background: #fce4ec;
        color: #c62828;
        padding: 4px 12px;
        border-radius: 16px;
        margin: 3px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-title">Fake News Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Paste a news article below and the model will tell you '
    'whether it looks <b>real</b> or <b>fake</b>.</div>',
    unsafe_allow_html=True,
)
st.divider()

news_input = st.text_area(
    "Enter the news headline or article text:",
    height=200,
    placeholder="e.g.  Breaking: Scientists discover high levels of lead in major water supply...",
)

analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

if analyze_btn:
    raw = news_input.strip()

    if not raw:
        st.warning("Please paste some text first.")
        st.stop()

    clean = preprocess_text(raw)
    X = tfidf_vec.transform([clean])
    prediction = svm_model.predict(X)[0]
    probas = svm_model.predict_proba(X)[0]

    confidence = float(max(probas)) * 100
    label = "Fake" if prediction == 1 else "Real"

    st.markdown("---")

    if label == "Real":
        st.markdown(
            f'<div class="result-real">'
            f'<h2 style="margin:0">Likely Real News</h2>'
            f'<p style="margin:0.5rem 0 0 0">Confidence: <b>{confidence:.1f}%</b></p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="result-fake">'
            f'<h2 style="margin:0">Likely Fake News</h2>'
            f'<p style="margin:0.5rem 0 0 0">Confidence: <b>{confidence:.1f}%</b></p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)
    col1.metric("Real probability", f"{probas[0] * 100:.1f}%")
    col2.metric("Fake probability", f"{probas[1] * 100:.1f}%")

    st.markdown("### Key words that influenced this prediction")

    keywords = get_top_keywords(clean)
    if keywords:
        chips_html = ""
        for word, weight, _ in keywords:
            css_class = "chip-fake" if weight > 0 else "chip-real"
            direction = "fake" if weight > 0 else "real"
            chips_html += f'<span class="{css_class}">{word} ({direction})</span> '
        st.markdown(chips_html, unsafe_allow_html=True)
    else:
        st.info("Not enough distinctive features found for this input.")

st.markdown("---")
st.caption(
    "Built with Streamlit  |  SVM + TF-IDF model trained on the WELFake dataset  |  "
    "AI/ML Project, Sem 4"
)
