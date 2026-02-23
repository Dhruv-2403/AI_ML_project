# Fake News Credibility Classifier

SVM Linear Model System to evaluate news article credibility whether it is Fake or Real trained on WELFake dataset .

## Overview

This project uses a Linear SVM model trained on the WELFake dataset to classify news articles as real or fake. The system combines machine learning with rule-based legitimacy checks for improved accuracy on current news.

## Model Performance (Evaluation Metrics)

The model was trained and evaluated on 20,000 articles with the following metrics:

| Metric | Value |
|--------|-------|
| **Accuracy** | 94.30% |
| **Precision** | 93.96% |
| **Recall** | 94.88% |
| **F1-Score** | 94.42% |

### Metric Definitions:
- **Accuracy**: Percentage of correctly classified articles (both real and fake)
- **Precision**: Of articles predicted as fake, how many were actually fake
- **Recall**: Of all actual fake articles, how many were correctly identified
- **F1-Score**: Harmonic mean of precision and recall

## Features

- Linear SVM classification model
- TF-IDF feature extraction (10,000 features)
- Legitimacy checks for news agencies and journalistic language
- Real-time credibility scoring (0-100)
- Confidence percentage display
- Current event detection (2024-2025 topics)

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   streamlit run app.py
   ```

3. **Open browser** at http://localhost:8501

## Project Structure

```
├── app.py              
├── model.ipynb        
├── requirements.txt    
├── svm_model.joblib    
├── tfidf_vectorizer.joblib 
└── .gitignore         
```

## Dataset

- **Source**: WELFake Dataset
- **Size**: 72,000+ news articles
- **Labels**: Real (0) and Fake (1)
- **Training Sample**: 20,000 articles for faster training

## How It Works

1. **Text Preprocessing**: Lowercasing, URL removal, punctuation removal, digit removal
2. **Feature Extraction**: TF-IDF vectorization with unigrams and bigrams
3. **Classification**: Linear SVM predicts real vs fake
4. **Legitimacy Checks**: Rule-based validation for:
   - News agency identifiers (Reuters, AP, etc.)
   - Journalistic language (said, according to, reported)
   - Current events (2024, 2025, Gaza, Ukraine, COVID, elections)

## Usage

1. Enter news article text in the text area (minimum 30 characters)
2. Click "Analyze" button
3. View results:
   - Classification: Real News or Fake News
   - Credibility Score: 0-100
   - Confidence: Percentage
   - Debug Info: Legitimacy and fake indicator counts

## Model Details

- **Algorithm**: Linear Support Vector Machine (SVM)
- **Features**: TF-IDF with 10,000 max features
- **Stop Words**: English
- **Training Time**: ~2-3 minutes (Trained on 20000 articles till now for Milestone-1 but our dataset contains 72K articles, we can further train more for our Milestone-2)

## Limitations

- Model trained on 2016-2018 data
- May not recognize very recent events or names
- Best suited for formal news articles
- Short texts (<30 chars) not supported

## Documentation & References

Our team has taken help from the documentation of scikit learn , streamlit ,etc for help if stuck anywhere. 

### Dataset Reference
- **WELFake Dataset**: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

### Library Documentation
- **Scikit-learn**: https://scikit-learn.org/stable/
- **Streamlit**: https://docs.streamlit.io/
- **Pandas**: https://pandas.pydata.org/docs/
- **NLTK**: https://www.nltk.org/
