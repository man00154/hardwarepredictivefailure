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

# ---------- Gemini Client with Retries ----------
def gemini_generate(api_key, prompt, max_output_tokens=400):
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}

    # safety: trim input if too long
    if len(prompt) > 4000:
        prompt = prompt[:4000]

    def call_gemini(tokens):
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": tokens}
        }
        try:
            resp = requests.post(API_URL, headers=headers, params=params, json=payload, timeout=60)
            data = resp.json()
            if "candidates" in data and data["candidates"]:
                cand = data["candidates"][0]
                if "content" in cand and "parts" in cand["content"]:
                    for part in cand["content"]["parts"]:
                        if "text" in part:
                            return part["text"].strip()
            return None
        except Exception:
            return None

    # Try with decreasing tokens
    for tokens in [max_output_tokens, 200, 100]:
        result = call_gemini(tokens)
        if result:
            return result

    return "⚠️ Gemini API returned no usable text, even after retries. Please try again with smaller input."

# ---------- Agentic Pipeline ----------
def agentic_pipeline(api_key, incident, rag):
    # Step 1: Hypotheses
    hypo = gemini_generate(api_key, f"List 3 possible hypotheses for this incident:\n{incident}", max_output_tokens=200)

    # Step 2: Retrieve context
    context = []
    for q in hypo.splitlines()[:3]:
        context.extend(rag.similarity_search(q, k=2))
    context_text = "\n---\n".join(context) if context else "No extra evidence."

    # Step 3: Final RCA
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
    return gemini_generate(api_key, final_prompt, max_output_tokens=400)

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
