# app.py
import os
import requests
import streamlit as st

# ---------- Gemini Config ----------
MODEL_NAME = "gemini-2.5-flash-preview-05-20"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

def get_api_key():
    return os.getenv("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY", ""))

# ---------- Tiny RAG (Mini Knowledge Base) ----------
class TinyRAG:
    def __init__(self):
        self.docs = {
            "disk": "Disk errors often predict HDD/SSD failure.",
            "temperature": "High temperature can cause CPU/GPU shutdown.",
            "memory": "Repeated memory errors may indicate failing RAM.",
            "power": "Power supply fluctuations can cause instability."
        }

    def retrieve(self, query):
        found = []
        for key, value in self.docs.items():
            if key in query.lower():
                found.append(value)
        return " ".join(found) if found else "No direct match, rely on AI."

# ---------- Gemini Client ----------
def gemini_generate(api_key, prompt, max_tokens=300):
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens}
    }
    try:
        resp = requests.post(API_URL, headers=headers, params=params, json=payload, timeout=30)
        data = resp.json()
        for cand in data.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                if "text" in part:
                    return part["text"].strip()
        return "⚠️ Gemini API returned no usable text."
    except Exception as e:
        return f"⚠️ Error: {e}"

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Predictive Hardware Failure RCA", layout="wide")
st.title("🔧 MANISH SINGH -  Predictive Hardware Failure RCA")

with st.sidebar:
    api_key = st.text_input("Enter GEMINI_API_KEY", type="password") or get_api_key()

incident_input = st.text_area("Paste a simple hardware log:", height=150)

if st.button("Run RCA"):
    if not api_key:
        st.error("Please provide GEMINI_API_KEY.")
    elif not incident_input.strip():
        st.warning("Please paste a log.")
    else:
        with st.spinner("Analyzing..."):
            # --- Tiny RAG retrieval ---
            rag = TinyRAG()
            context = rag.retrieve(incident_input)

            # --- Agentic AI Prompt ---
            prompt = f"""
You are an expert Data Center Engineer for predictive hardware failure analysis. 
Use the retrieved knowledge and log to provide a predictive RCA.

Knowledge Base: {context}

Log: {incident_input}

Give a short structured report:
1. Root Cause
2. Prediction
3. Immediate Fix
4. Long-term Solution
"""
            report = gemini_generate(api_key, prompt, max_tokens=250)

        st.subheader("📄 Predictive RCA Report")
        st.markdown(report)
