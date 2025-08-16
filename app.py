# app.py
import os
import io
import json
import requests
import streamlit as st
from typing import List
from PyPDF2 import PdfReader

# ---------- Gemini Config ----------
MODEL_NAME = "gemini-2.5-flash-preview-05-20"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

def get_api_key() -> str:
    return os.getenv("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY", ""))

# ---------- Helpers ----------
def chunk_text(text: str, max_chars: int = 800, overlap: int = 50) -> List[str]:
    text = text.replace("\x00", "")
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks

# ---------- Simple RAG ----------
class SimpleRAG:
    def __init__(self):
        self.docs = []

    def add_documents(self, texts: List[str]):
        for t in texts:
            self.docs.append(t)

    def similarity_search(self, query: str, k: int = 3):
        q = query.lower().split()
        scored = []
        for d in self.docs:
            score = sum(1 for w in q if w in d.lower())
            if score > 0:
                scored.append((score, d))
        scored.sort(key=lambda x: -x[0])
        return [d for _, d in scored[:k]]

# ---------- Gemini Client ----------
def gemini_generate(api_key: str, user_prompt: str, system_prompt: str = "", max_output_tokens: int = 800) -> str:
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": f"{system_prompt}\n{user_prompt}"}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_output_tokens}
    }
    resp = requests.post(API_URL, headers=headers, params=params, json=payload, timeout=60)
    if resp.status_code != 200:
        return f"API Error {resp.status_code}: {resp.text}"
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return "⚠️ No valid response from Gemini."

# ---------- Agentic Pipeline ----------
def agentic_pipeline(api_key: str, incident: str, rag: SimpleRAG) -> str:
    # Step 1: Hypotheses
    hypo = gemini_generate(api_key, f"Generate 3 short hypotheses for this incident:\n{incident}")
    
    # Step 2: Retrieve context
    queries = hypo.splitlines()[:3]
    context = []
    for q in queries:
        context.extend(rag.similarity_search(q, k=2))
    context_text = "\n---\n".join(context) if context else "[No context found]"
    
    # Step 3: Final Analysis
    final_prompt = f"""
Incident:
{incident}

Hypotheses:
{hypo}

Evidence:
{context_text}

Write a predictive RCA with:
1. Root Cause
2. Prediction
3. Evidence
4. Immediate Actions
5. Long-term Remediation
6. Final Solution Summary
"""
    return gemini_generate(api_key, final_prompt)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Predictive Hardware Failure Analysis", layout="wide")
st.title("🔧 MANISH SINGH - Predictive Hardware Failure Analysis (Simplified)")

with st.sidebar:
    api_key = st.text_input("Enter GEMINI_API_KEY", type="password") or get_api_key()
    uploaded_files = st.file_uploader("Upload logs or PDFs", accept_multiple_files=True, type=["txt", "log", "pdf"])

# Build knowledge base
rag = SimpleRAG()
kb_texts = []
if uploaded_files:
    for uf in uploaded_files:
        if uf.name.lower().endswith((".txt", ".log")):
            kb_texts.extend(chunk_text(uf.read().decode("utf-8", errors="ignore")))
        elif uf.name.lower().endswith(".pdf"):
            reader = PdfReader(io.BytesIO(uf.read()))
            txt = "\n".join([p.extract_text() or "" for p in reader.pages])
            kb_texts.extend(chunk_text(txt))
if kb_texts:
    rag.add_documents(kb_texts)
    st.success(f"KB ready with {len(kb_texts)} chunks.")

# Incident input
incident_input = st.text_area("Paste telemetry/logs:", height=200)

# Run analysis
if st.button("Run Analysis"):
    if not api_key:
        st.error("Please provide GEMINI_API_KEY.")
    elif not incident_input.strip():
        st.warning("Please paste telemetry.")
    else:
        with st.spinner("Running predictive analysis..."):
            report = agentic_pipeline(api_key, incident_input.strip(), rag)
        st.subheader("📄 Predictive RCA Report")
        st.markdown(report)

# Example
if st.button("Load Example Incident"):
    st.session_state["incident_example"] = (
        "Node: server-23\n"
        "Metric: disk latency very high, SMART errors detected\n"
        "Reallocated sector count increasing rapidly\n"
        "Recent firmware update applied\n"
    )
    st.experimental_rerun()
