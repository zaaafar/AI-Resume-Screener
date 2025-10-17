"""
Simple model utilities: train and predict.
This trains a TF-IDF + LogisticRegression pipeline on the sample CSV
and saves the model to disk as `model.joblib`.
"""

import os
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Define paths
MODEL_PATH = Path(__file__).parent / "model.joblib"
DATA_PATH = Path(__file__).parent / "sample_data.csv"

def train_and_save(retrain: bool = False):
    """
    Train model from sample CSV and save.
    If model exists and retrain=False, skip training.
    """
    if MODEL_PATH.exists() and not retrain:
        return str(MODEL_PATH)

    # Load dataset
    df = pd.read_csv(DATA_PATH)
    X = df['text'].fillna("")
    y = df['label']

    # Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define the pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ('clf', LogisticRegression(max_iter=200))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    preds = pipeline.predict(X_test)
    print("\nClassification report (demo):\n", classification_report(y_test, preds))

    # Save the trained model
    joblib.dump(pipeline, MODEL_PATH)

    return str(MODEL_PATH)

def load_model():
    """
    Load the trained model from disk, training if necessary.
    """
    if not MODEL_PATH.exists():
        train_and_save()
    return joblib.load(MODEL_PATH)

def predict(texts):
    """
    Predict labels and probabilities for given list of texts.
    """
    model = load_model()
    return model.predict(texts).tolist(), model.predict_proba(texts).tolist()