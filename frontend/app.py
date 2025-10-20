import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO
import base64
import json

# =========================
# CONFIG & STYLING
# =========================
st.set_page_config(page_title="AI Resume Screener", layout='wide', initial_sidebar_state='expanded')

def set_dark_theme():
    st.markdown("""
        <style>
        .stApp {background-color: #0f1724; color: #e6eef8; font-family: 'Segoe UI', sans-serif;}
        .stButton>button {background-color: #111827; color: #e6eef8; border: 1px solid #444; padding: 0.6rem 1.2rem; border-radius: 6px;}
        .stButton>button:hover {background-color: #1f2937;}
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {background-color:#0b1220; color:#e6eef8; border: 1px solid #333; padding: 0.5rem; border-radius: 6px;}
        .stFileUploader>div>div {background-color:#0b1220; color:#e6eef8; border: 1px solid #333;}
        .card {background-color: #111827; padding: 1.5rem; margin-bottom: 1rem; border-radius: 10px; border: 1px solid #333; box-shadow: 0 4px 6px rgba(0,0,0,0.3);}
        .skill-tag {background-color:#34d399; color:#0f1724; padding: 5px 12px; border-radius: 20px; margin: 3px; display: inline-block; font-size: 0.85rem; font-weight: 500;}
        .metric-box {background: linear-gradient(135deg, #34d399 0%, #10b981 100%); color: #0f1724; padding: 1rem; border-radius: 10px; text-align: center; margin: 0.5rem;}
        .metric-value {font-size: 2rem; font-weight: bold;}
        .metric-label {font-size: 0.9rem; margin-top: 0.5rem;}
        .role-match {background-color: #1f2937; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #34d399;}
        .role-match.low {border-left-color: #ef4444;}
        .contact-info {background-color: #0b1220; padding: 0.8rem; border-radius: 6px; margin: 0.3rem 0; border-left: 3px solid #3b82f6;}
        footer {visibility: hidden}
        </style>
    """, unsafe_allow_html=True)

set_dark_theme()

# API configuration
API_BASE = "http://localhost:8000"

# =========================
# HEADER & SIDEBAR
# =========================
def render_header():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🚀 AI Resume Screener")
        st.markdown("*Intelligent resume analysis powered by machine learning*")
    with col2:
        if st.button("🔄 Retrain Model", key="retrain_btn"):
            try:
                resp = requests.post(f"{API_BASE}/retrain")
                if resp.status_code == 200:
                    st.success("✅ Model retrained successfully!")
            except Exception as e:
                st.error(f"Retrain failed: {e}")

render_header()

# Sidebar
with st.sidebar:
    st.header("📋 About")
    st.markdown("""
    This AI Resume Screener uses machine learning to:
    - **Classify** resumes into job roles
    - **Extract** technical skills and experience
    - **Analyze** career level and qualifications
    - **Match** resume content to role requirements
    
    **Supported Formats:** PDF, DOCX, TXT
    """)
    
    st.divider()
    st.header("⚙️ Model Info")
    if st.button("🏥 Health Check"):
        try:
            resp = requests.get(f"{API_BASE}/health")
            if resp.status_code == 200:
                st.success("✅ API is healthy!")
        except:
            st.error("❌ API is down!")

# =========================
# UPLOAD COMPONENT
# =========================
def get_resume_input():
    st.subheader("📄 Upload or Paste Resume")
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_files = st.file_uploader(
            "Upload resume(s) (PDF/DOCX/TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )
    
    with col2:
        use_text = st.checkbox("Or paste text instead")
    
    raw_text = ""
    if use_text:
        raw_text = st.text_area("Paste resume text here", height=200)
    
    return uploaded_files, raw_text

uploaded_files, raw_text = get_resume_input()

# =========================
# PREDICTION & ANALYSIS
# =========================
def analyze_resume(uploaded_files, raw_text):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        analyze_btn = st.button("📊 Analyze Resume(s)", key="analyze_btn")
    with col2:
        if st.button("📥 Download Results", key="download_btn"):
            analyze_btn = True
    
    if analyze_btn:
        if not uploaded_files and not raw_text.strip():
            st.error("⚠️ Please upload a file or paste text to analyze.")
            return

        results = []
        progress_bar = st.progress(0)
        
        try:
            files_to_process = uploaded_files if uploaded_files else [None]
            
            for idx, uploaded in enumerate(files_to_process):
                progress_bar.progress((idx + 1) / len(files_to_process))
                
                files = {}
                data = {}
                
                if uploaded:
                    files = {"file": (uploaded.name, uploaded.getvalue())}
                    resp = requests.post(f"{API_BASE}/predict", files=files)
                    source_name = uploaded.name
                else:
                    data = {"text": raw_text}
                    resp = requests.post(f"{API_BASE}/predict", data=data)
                    source_name = "📝 Pasted Text"

                if resp.status_code == 200:
                    result = resp.json()
                    result['source_name'] = source_name
                    results.append(result)
                else:
                    st.error(f"❌ Analysis failed for {source_name}: {resp.status_code}")
            
            progress_bar.empty()
            
            if results:
                st.success(f"✅ Analyzed {len(results)} resume(s)")
                st.divider()
                
                for result in results:
                    render_detailed_analysis(result)
                
        except Exception as e:
            st.error(f"❌ Request failed: {e}")

def render_detailed_analysis(result):
    """Render comprehensive resume analysis."""
    source = result.get('source_name', 'Unknown')
    st.markdown(f"### {source}")
    
    # Top metrics row - only predicted role
    predicted_role = result.get('predictions', ['Unknown'])[0]
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #34d399 0%, #10b981 100%); color: #0f1724; padding: 2rem; border-radius: 15px; text-align: center; box-shadow: 0 8px 16px rgba(52, 211, 153, 0.3);'>
        <div style='font-size: 1.2rem; margin-bottom: 0.5rem; opacity: 0.9;'>Predicted Role</div>
        <div style='font-size: 3rem; font-weight: bold;'>{predicted_role}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    

    
    # Role confidence scores - full width
    st.subheader("📊 Role Compatibility")
    role_scores = result.get('role_scores', {})
    
    if role_scores:
        # Create bar chart
        roles = list(role_scores.keys())
        scores = [role_scores[r] for r in roles]
        
        fig = go.Figure(
            data=[go.Bar(
                x=roles,
                y=scores,
                marker=dict(
                    color=scores,
                    colorscale='Greens',
                    showscale=False
                ),
                text=[f"{s:.1%}" for s in scores],
                textposition="outside"
            )]
        )
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='#111827',
            plot_bgcolor='#0b1220',
            font=dict(color='#e6eef8'),
            yaxis=dict(range=[0, 1]),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()

def download_results(results):
    """Generate downloadable CSV report."""
    try:
        data = []
        for result in results:
            data.append({
                'Source': result.get('source_name', 'Unknown'),
                'Predicted Role': result.get('predictions', ['Unknown'])[0],
                'Confidence': f"{result.get('confidence', 0):.2%}",
                'Match Score': f"{result.get('match_score', 0):.2%}",
                'Experience Level': result.get('experience_level', 'Unknown'),
                'Skills': ', '.join(result.get('skills', [])[:10]),
                'Education': ', '.join(result.get('education', [])),
            })
        
        df = pd.DataFrame(data)
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        
        st.markdown(
            f'<a href="data:file/csv;base64,{b64}" download="resume_analysis.csv" style="color: #34d399;">💾 Download Results CSV</a>',
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Failed to generate report: {e}")

# =========================
# RUN APP
# =========================
analyze_resume(uploaded_files, raw_text)