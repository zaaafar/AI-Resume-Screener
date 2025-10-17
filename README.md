# AI-Resume-Screener (Local Demo)


Simple full-stack demo: Streamlit frontend + FastAPI backend. The backend trains a small TF-IDF + LogisticRegression classifier from `backend/sample_data.csv` and exposes `/predict` and `/retrain` endpoints. The frontend allows uploading resumes (PDF/DOCX) or pasting text.


## Setup (Linux/Mac/Windows)


1. Create a folder and paste files as shown.
2. Create virtual environment (recommended):


```bash
python -m venv .venv
source .venv/bin/activate # (Windows: .venv\Scripts\activate)