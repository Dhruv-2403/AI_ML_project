# Intelligent News Credibility Analysis

This project is an AI-driven content analysis system designed to assess the credibility of news articles( Fake / Real ) using machine learning and textual features. 

## Milestone 1 Overview

This milestone focuses on building the foundational Machine Learning pipeline and a basic user interface for news credibility classification.

### Flow of the Project

1. **Data Loading and Preprocessing**: 
   - The system loads a sample of the dataset (e.g., WELFake Dataset).


   - Text is preprocessed to remove punctuation, convert to lowercase, and neutralize non-alphanumeric characters.
   
2. **Feature Extraction**:
   - The cleaned text is converted into a numerical feature matrix using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.

3. **Model Training**:
   - A Logistic Regression model is trained on the TF-IDF features to classify articles as either "Real" or "Fake".

   - The model and vectorizer are evaluated and saved as joblib files for later use in the prediction application.

4. **Prediction / Inference UI**:
   - A user interface built with Streamlit allows users to input news text.

   - The input is preprocessed, vectorized, and passed to the trained model.

   - The app applies custom legitimacy heuristics and computes a final credibility score and confidence percentage before displaying the result to the user.

## Project Structure

- `config/`: Contains configuration settings and environment variables.
- `models/`: Where the trained models (`lr_model.joblib`, `tfidf_vectorizer.joblib`) are saved and loaded from.
- `src/train.py`: The script to train the model and generate the joblib artifacts.
- `src/app.py`: The Streamlit web application.
- `utils/`: Helper modules for data loading, preprocessing, and feature extraction.

## How to Run the Project

Follow these commands in your terminal to set up and run the system locally:

### 1. Install Dependencies
Ensure you have Python installed, then install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Prepare the Data
Download the [WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) from Kaggle.
Ensure the dataset (`WELFake_Dataset.csv`) is placed in the `data/` directory.

### 3. Training the Model
Run the training script to generate the necessary model files. This script will train the Logistic Regression model, evaluate it, and save it to the `models/` directory.
```bash
python src/train.py
```

### 4. Run the Streamlit Application
Once the model is trained and the `.joblib` files are saved, you can launch the web interface:
```bash
streamlit run src/app.py
```

This command will start a local web server where you can interact with the credibility checker.
