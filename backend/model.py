import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Define paths
MODEL_PATH = Path(__file__).parent / "model.joblib"
DATA_PATH = Path(__file__).parent / "sample_data.csv"

def clean_text(text):
    """
    Advanced text cleaning for better model performance.
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep spaces and alphanumeric
    text = re.sub(r'[^a-zA-Z0-9\s+#]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_keywords(text):
    """
    Extract domain-specific keywords for feature enhancement.
    """
    keywords_map = {
        'backend': ['backend', 'server', 'api', 'database', 'sql', 'node', 'express', 'django', 'flask', 'java', 'spring'],
        'frontend': ['frontend', 'react', 'vue', 'angular', 'javascript', 'css', 'html', 'ui', 'ux', 'responsive'],
        'data_scientist': ['data', 'machine learning', 'ml', 'python', 'pandas', 'numpy', 'sklearn', 'tensorflow', 'pytorch', 'analytics'],
        'devops': ['devops', 'docker', 'kubernetes', 'ci/cd', 'jenkins', 'aws', 'azure', 'gcp', 'terraform', 'ansible', 'linux']
    }
    
    text_lower = text.lower()
    keyword_counts = {role: 0 for role in keywords_map}
    
    for role, keywords in keywords_map.items():
        for keyword in keywords:
            keyword_counts[role] += text_lower.count(keyword)
    
    return keyword_counts

def create_enhanced_features(df):
    """
    Create engineered features for improved predictions.
    """
    df['text_length'] = df['text'].fillna("").apply(len)
    df['word_count'] = df['text'].fillna("").apply(lambda x: len(x.split()))
    df['has_github'] = df['text'].fillna("").apply(lambda x: int('github' in x.lower()))
    df['has_portfolio'] = df['text'].fillna("").apply(lambda x: int('portfolio' in x.lower()))
    
    return df

def train_and_save(retrain: bool = False):
    """
    Train enhanced model from sample CSV with hyperparameter tuning.
    """
    if MODEL_PATH.exists() and not retrain:
        return str(MODEL_PATH)

    # Load dataset
    df = pd.read_csv(DATA_PATH)
    df['text'] = df['text'].apply(clean_text)
    df = create_enhanced_features(df)
    
    X = df['text'].fillna("")
    y = df['label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define enhanced pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=10000,
            min_df=2,
            max_df=0.8,
            sublinear_tf=True,
            stop_words='english'
        )),
        ('clf', LinearSVC(max_iter=2000, random_state=42, dual=False))
    ])

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'tfidf__ngram_range': [(1, 2), (1, 3)],
        'tfidf__max_features': [5000, 10000],
        'clf__C': [0.1, 1.0, 10.0],
    }
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )

    # Train with cross-validation
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    print(f"\n✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best CV score: {grid_search.best_score_:.4f}")

    # Evaluate
    preds = best_model.predict(X_test)
    print("\n📊 Classification Report:\n", classification_report(y_test, preds))
    print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, preds))
    
    test_f1 = f1_score(y_test, preds, average='weighted')
    print(f"\n📈 Test F1 Score: {test_f1:.4f}")

    # Save model
    joblib.dump(best_model, MODEL_PATH)
    return str(MODEL_PATH)

def load_model():
    """
    Load trained model or train if not exists.
    """
    if not MODEL_PATH.exists():
        train_and_save()
    return joblib.load(MODEL_PATH)

def predict(texts):
    """
    Predict labels and probabilities for given texts.
    """
    model = load_model()
    
    # Clean texts
    texts = [clean_text(t) for t in texts]
    
    predictions = model.predict(texts).tolist()
    
    # Get decision scores and convert to probabilities
    decision_scores = model.decision_function(texts)
    
    # Convert to probability-like scores using softmax
    if decision_scores.ndim == 1:
        decision_scores = decision_scores.reshape(-1, 1)
    
    exp_scores = np.exp(decision_scores - decision_scores.max(axis=1, keepdims=True))
    probabilities = (exp_scores / exp_scores.sum(axis=1, keepdims=True)).tolist()
    
    return predictions, probabilities