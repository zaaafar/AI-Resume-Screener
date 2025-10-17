from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import io
from pathlib import Path
import docx2txt
import pdfplumber
import uvicorn
from .model import train_and_save, predict

app = FastAPI(title="AI Resume Screener - Backend")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class PredictRequest(BaseModel):
    text: List[str]

# Startup event to ensure model exists
@app.on_event("startup")
def startup_event():
    # ensure model exists
    train_and_save()

# Prediction endpoint
@app.post("/predict")
async def predict_endpoint(
    text: str = Form(None),
    file: UploadFile = File(None)
):
    """Accepts either raw text (form field `text`) or a file upload `file` (pdf/docx).
    Returns predicted role and probabilities."""
    content_text = ""

    if file is not None:
        filename = file.filename.lower() # type: ignore
        data = await file.read()

        if filename.endswith(".pdf"):
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
                content_text = "\n".join(pages)
        elif filename.endswith(('.docx', '.doc')):
            # write to temp and use docx2txt
            tmp = Path("temp_upload.docx")
            tmp.write_bytes(data)
            content_text = docx2txt.process(str(tmp))
            tmp.unlink(missing_ok=True)
        else:
            # assume plain text
            content_text = data.decode('utf-8', errors='ignore')
    elif text is not None:
        content_text = text
    else:
        return {"error": "Provide either `text` or upload a file."}

    texts = [content_text]
    labels, probs = predict(texts)
    return {"predictions": labels, "probabilities": probs}

# Retraining endpoint
@app.post("/retrain")
async def retrain_endpoint():
    path = train_and_save(retrain=True)
    return {"status": "retrained", "model_path": path}

# Run the app
if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)