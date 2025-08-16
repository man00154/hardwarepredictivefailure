# app.py
import os
import io
import json
import asyncio
import aiohttp
import streamlit as st
from typing import List, Dict, Any, Optional
from PyPDF2 import PdfReader

# ---------- Gemini config ----------
MODEL_NAME = "gemini-2.5-flash-preview-05-20"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

# ---------- Helpers ----------
def get_api_key() -> str:
    key = None
    if hasattr(st, "secrets"):
        key = st.secrets.get("GEMINI_API_KEY")
    if not key:
        key = os.getenv("GEMINI_API_KEY")
    return key or ""

def try_run(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    else:
        return asyncio.run(coro)

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 100) -> List[str]:
    text = text.replace("\x00", "")
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def safe_trim(text: str, max_chars: int):
    if not text:
        return ""
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]"

# ---------- RAG ----------
class SimpleRAG:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.docs: List[Dict[str, Any]] = []
        self.emb_model = None
        self.index = None
        self.using_faiss = False
        try:
            import sentence_transformers  # noqa
            import faiss  # noqa
            self.using_faiss = True
        except Exception:
            self.using_faiss = False

    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        if metadatas is None:
            metadatas = [{} for _ in texts]
        for t, m in zip(texts, metadatas):
            self.docs.append({"text": t, "metadata": m})

    def build(self):
        if not self.docs or not self.using_faiss:
            return
        from sentence_transformers import SentenceTransformer
        import numpy as np, faiss
        self.emb_model = SentenceTransformer(self.model_name)
        corpus = [d["text"] for d in self.docs]
        embeddings = self.emb_model.encode(
            corpus, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
        )
        dim = int(embeddings.shape[1])
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype("float32"))

    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.using_faiss and self.index is not None:
            q_emb = self.emb_model.encode(
                [query], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
            ).astype("float32")
            D, I = self.index.search(q_emb, k)
            results = []
            for score, idx in zip(D[0], I[0]):
                if idx == -1:
                    continue
                results.append({
                    "text": self.docs[int(idx)]["text"],
                    "metadata": self.docs[int(idx)].get("metadata", {}),
                    "score": float(score),
                })
            return results
        q = query.lower().split()
        scored = []
        for d in self.docs:
            text = d["text"].lower()
            score = sum(1 for w in q if w and w in text)
            if score > 0:
                scored.append((score, d))
        scored.sort(key=lambda x: -x[0])
        return [{"text": d["text"], "metadata": d.get("metadata", {}), "score": float(s)} for s, d in scored[:k]]

# ---------- Gemini client ----------
async def gemini_generate_async(api_key: str, user_prompt: str, system_prompt: Optional[str] = None,
                                temperature: float = 0.2, max_output_tokens: int = 1200, retries: int = 2) -> str:
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    concise_guardrail = (
        "Be structured. Always include explicit sections:\n"
        "1. Root Cause\n2. Prediction (Future Risk)\n3. Evidence (from logs/context)\n"
        "4. Immediate Actions\n5. Long-term Remediation\n6. Final Solution Summary"
    )

    for attempt in range(retries + 1):
        effective_max = max(256, int(max_output_tokens * (0.7 ** attempt)))
        parts = []
        if system_prompt:
            parts.append({"text": f"[SYSTEM]\n{system_prompt}\n\n{concise_guardrail}"})
        else:
            parts.append({"text": concise_guardrail})
        parts.append({"text": user_prompt})
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": effective_max,
                "responseMimeType": "text/plain"
            }
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(API_URL, headers=headers, params=params, json=payload, timeout=120) as resp:
                    text_body = await resp.text()
                    if resp.status != 200:
                        if resp.status in (429, 500, 502, 503, 504) and attempt < retries:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        raise RuntimeError(f"Gemini API error {resp.status}: {text_body}")
                    data = json.loads(text_body)
                    try:
                        candidate = data.get("candidates", [{}])[0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            for part in candidate["content"]["parts"]:
                                if "text" in part and part["text"].strip():
                                    return part["text"].strip()
                    except Exception:
                        pass
                    finish_reason = data.get("candidates", [{}])[0].get("finishReason", "")
                    if finish_reason == "MAX_TOKENS" and attempt < retries:
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    return "No text generated (likely due to token limit)."
        except aiohttp.ClientError as e:
            if attempt < retries:
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            return f"Network error: {e}"
    return "Failed to get response from Gemini API."

def gemini_generate(*args, **kwargs) -> str:
    return try_run(gemini_generate_async(*args, **kwargs))

# ---------- Agentic pipeline ----------
def agentic_predictive_pipeline(api_key: str, incident: str, rag: SimpleRAG, temperature: float = 0.2) -> str:
    hypo_prompt = f"You are a data center engineer. Give 3 short hypotheses for the incident:\n{incident}"
    hypotheses = gemini_generate(
        api_key=api_key,
        user_prompt=hypo_prompt,
        system_prompt="Generate 3 short hypotheses.",
        temperature=temperature,
        max_output_tokens=200,
    )

    qry_prompt = f"From these hypotheses, make up to 5 search queries (one per line):\n{hypotheses}"
    queries_text = gemini_generate(
        api_key=api_key,
        user_prompt=qry_prompt,
        system_prompt="Output queries only.",
        temperature=0.1,
        max_output_tokens=200,
    )
    queries = [q.strip() for q in queries_text.splitlines() if q.strip()][:5]

    retrieved = []
    for q in queries:
        hits = rag.similarity_search(q, k=3)
        for h in hits:
            retrieved.append(h["text"])
    context_text = safe_trim("\n---\n".join(dict.fromkeys(retrieved)), 4000)

    final_prompt = f"""
Incident:
{incident}

Hypotheses:
{hypotheses}

Evidence (retrieved context):
{context_text if context_text else '[no retrieved context]'}

Write a predictive analysis including:
1) Root Cause
2) Prediction (future risk if unaddressed)
3) Evidence (from logs/context)
4) Immediate Actions
5) Long-term Remediation
6) Final Solution Summary (clear resolution in 2–3 sentences)
"""
    return gemini_generate(
        api_key=api_key,
        user_prompt=final_prompt,
        system_prompt="Deliver predictive RCA and a solution summary with actionable steps.",
        temperature=temperature,
        max_output_tokens=900,
    )

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Predictive Hardware Failure Analysis", layout="wide")
st.title("🔧 MANISH SINGH - Predictive Hardware Failure Analysis (Gemini + RAG + Agent)")

with st.sidebar:
    manual_key = st.text_input("Enter GEMINI_API_KEY (optional)", type="password", key="api_key_input")
    api_key = manual_key or get_api_key()
    if not api_key:
        st.warning("Set GEMINI_API_KEY in secrets or environment.")
    uploaded_files = st.file_uploader("Upload .txt/.log/.pdf", accept_multiple_files=True, type=["txt", "log", "pdf"])

kb_texts: List[str] = []
if uploaded_files:
    for uf in uploaded_files:
        try:
            if uf.name.lower().endswith((".txt", ".log")):
                kb_texts.extend(chunk_text(uf.read().decode("utf-8", errors="ignore")))
            elif uf.name.lower().endswith(".pdf"):
                reader = PdfReader(io.BytesIO(uf.read()))
                txt = "\n".join([p.extract_text() or "" for p in reader.pages])
                kb_texts.extend(chunk_text(txt))
        except Exception as e:
            st.warning(f"Failed to read {uf.name}: {e}")

rag = SimpleRAG()
if kb_texts:
    rag.add_documents(kb_texts, [{"source": "upload"} for _ in kb_texts])
    try:
        rag.build()
        st.success(f"KB ready with {len(kb_texts)} chunks.")
    except Exception as e:
        st.warning(f"RAG build failed: {e}. Using keyword fallback.")

# --- Initialize session state safely ---
if "incident_input" not in st.session_state:
    st.session_state["incident_input"] = ""

incident_input = st.text_area("Paste telemetry/logs:", height=240, key="incident_input")

# --- RUN ANALYSIS ---
if st.button("Run Predictive Analysis"):
    if not api_key:
        st.error("Gemini API key required.")
    elif not incident_input.strip():
        st.warning("Please paste telemetry.")
    else:
        with st.spinner("Running predictive analysis..."):
            try:
                report = agentic_predictive_pipeline(api_key=api_key, incident=incident_input.strip(), rag=rag)
            except Exception as e:
                report = f"Error: {e}"
        st.subheader("📄 Predictive Hardware Failure Analysis")
        st.markdown(report)

# --- LOAD EXAMPLE ---
if st.button("Load example incident"):
    st.session_state["incident_input"] = (
        "Node: server-23\n"
        "Metric: disk_read_latency_ms=45 (baseline 5ms), disk_write_latency_ms=80 (baseline 7ms)\n"
        "SMART: reallocated_sector_count=120, current_pending_sector=4\n"
        "Event: repeated soft ECC errors reported on channel A\n"
        "Recent change: firmware update to storage controller 2 days ago\n"
    )
    st.rerun()
