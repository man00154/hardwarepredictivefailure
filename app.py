# app.py
import os
import io
import requests
import streamlit as st
from PyPDF2 import PdfReader

# ---------- Gemini Config ----------
MODEL_NAME = "gemini-2.5-flash-preview-05-20"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

def get_api_key():
    return os.getenv("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY", ""))

# ---------- Lightweight RAG ----------
class SimpleRAG:
    def __init__(self):
        self.docs = []

    def add_documents(self, texts):
        self.docs.extend(texts)

    def similarity_search(self, query, k=3):
        q_words = query.lower().split()
        scored = []
        for d in self.docs:
            score = sum(1 for w in q_words if w in d.lower())
            if score > 0:
                scored.append((score, d))
        scored.sort(key=lambda x: -x[0])
        return [d for _, d in scored[:k]]

# ---------- Gemini Client ----------
def gemini_generate(api_key, prompt, max_output_tokens=500):
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_output_tokens}
    }
    try:
        resp = requests.post(API_URL, headers=headers, params=params, json=payload, timeout=60)
        data = resp.json()

        if "candidates" in data and data["candidates"]:
            cand = data["candidates"][0]
            if "content" in cand and "parts" in cand["content"]:
                parts = cand["content"]["parts"]
                if parts and "text" in parts[0]:
                    return parts[0]["text"].strip()

        return "⚠️ Gemini API returned no usable text. Try reducing input size or tokens."
    except Exception as e:
        return f"⚠️ No valid response from Gemini: {str(e)}"

# ---------- Agentic Pipeline ----------
def agentic_pipeline(api_key, incident, rag):
    hypo = gemini_generate(api_key, f"List 3 possible hypotheses for this incident:\n{incident}")

    context = []
    for q in hypo.splitlines()[:3]:
        context.extend(rag.similarity_search(q, k=2))
    context_text = "\n---\n".join(context) if context else "No extra evidence."

    final_prompt = f"""
Incident:
{incident}

Hypotheses:
{hypo}

Evidence:
{context_text}

Write a predictive RCA report including:
1. Root Cause
2. Prediction
3. Evidence
4. Immediate Actions
5. Long-term Remediation
6. Final Solution Summary
"""
    return gemini_generate(api_key, final_prompt)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Predictive Hardware Failure RCA", layout="wide")
st.title("🔧 MANISH SINGH - Predictive Hardware Failure RCA (LLM + RAG + Agentic AI)")

with st.sidebar:
    api_key = st.text_input("Enter GEMINI_API_KEY", type="password") or get_api_key()
    uploaded_files = st.file_uploader("Upload logs/PDFs", accept_multiple_files=True, type=["txt", "log", "pdf"])

# Build KB
rag = SimpleRAG()
if uploaded_files:
    for uf in uploaded_files:
        if uf.name.endswith((".txt", ".log")):
            rag.add_documents([uf.read().decode("utf-8", errors="ignore")])
        elif uf.name.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(uf.read()))
            text = "\n".join([p.extract_text() or "" for p in reader.pages])
            rag.add_documents([text])
    st.success(f"Knowledge base ready with {len(rag.docs)} documents.")

# Input
incident_input = st.text_area("Paste telemetry/logs:", height=200)

# Run Analysis
if st.button("Run Analysis"):
    if not api_key:
        st.error("Please provide GEMINI_API_KEY.")
    elif not incident_input.strip():
        st.warning("Please paste telemetry.")
    else:
        with st.spinner("Running predictive RCA..."):
            report = agentic_pipeline(api_key, incident_input.strip(), rag)
        st.subheader("📄 Predictive RCA Report")
        st.markdown(report)
