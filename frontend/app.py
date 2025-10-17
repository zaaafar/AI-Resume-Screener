import streamlit as st
import requests
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="AI Resume Screener", layout='wide')

# Dark sleek CSS styling
st.markdown(
    """
    <style>
    .stApp {background-color: #0f1724; color: #e6eef8;}
    .css-1d391kg {background-color: #0b1220; color: #e6eef8;}
    .stButton>button {background-color: #111827; color: #e6eef8; border: 1px solid #444;}
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {background-color:#0b1220; color:#e6eef8;}
    .stFileUploader>div>div {background-color:#0b1220; color:#e6eef8;}
    .stMarkdown {text-align: justify;}
    footer {visibility: hidden}
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and description
st.title("🚀 AI-Powered Resume Screening")
st.markdown("""
This tool classifies resumes into job roles using a demo AI model.
Upload a resume (PDF/DOCX) or paste resume text below. 
The model will predict the most suitable role and show confidence scores.
""")

# Layout: two columns
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Input Resume")
    uploaded = st.file_uploader("Upload resume (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])
    raw_text = st.text_area("Or paste resume text here", height=300)
    submitted = st.button("Classify Resume")

with col2:
    st.subheader("Prediction Results")
    result_placeholder = st.empty()

# When submitted
if submitted:
    if not uploaded and not raw_text.strip():
        st.error("⚠️ Please upload a file or paste text to classify.")
    else:
        try:
            files = {}
            data = {"text": raw_text} if raw_text.strip() else None
            if uploaded:
                files = {"file": (uploaded.name, uploaded.getvalue())}
                resp = requests.post("http://localhost:8000/predict", files=files)
            else:
                resp = requests.post("http://localhost:8000/predict", data={"text": raw_text})

            if resp.status_code == 200:
                j = resp.json()
                preds = j.get("predictions", [])
                probs = j.get("probabilities", [])

                for i, p in enumerate(preds):
                    role = p
                    probability_list = probs[i]
                    # Show prediction prominently
                    result_placeholder.markdown(f"### ✅ Predicted Role: {role}")

                    # Convert probabilities to dataframe for display
                    if probability_list:
                        classes = ["Backend Engineer", "Data Scientist", "Frontend Engineer", "DevOps Engineer"]
                        df = pd.DataFrame({
                            "Role": classes,
                            "Probability": probability_list
                        })
                        df = df.sort_values("Probability", ascending=False)
                        st.markdown("#### Model Confidence Scores:")
                        st.table(df.style.format({"Probability": "{:.2%}"}))

                        # Optional: bar chart
                        st.bar_chart(df.set_index("Role"))

            else:
                st.error(f"Server error: {resp.status_code} - {resp.text}")

        except Exception as e:
            st.error(f"Request failed: {e}")
