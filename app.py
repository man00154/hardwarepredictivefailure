# app.py
import os
import io
import json
import asyncio
import aiohttp
import streamlit as st
from typing import List, Dict, Any, Optional

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

# ---------- RAG (tries to use sentence-transformers + faiss, else fallback) ----------
class SimpleRAG:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.docs: List[Dict[str, Any]] = []
        self.emb_model = None
        self.index = None
        self.using_faiss = False

        # try imports lazily
        try:
            import sentence_transformers  # noqa: F401
            import faiss  # noqa: F401
            self.using_faiss = True
        except Exception:
            self.using_faiss = False

    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        if metadatas is None:
            metadatas = [{} for _ in texts]
        for t, m in zip(texts, metadatas):
            self.docs.append({"text": t, "metadata": m})

    def build(self):
        if not self.docs:
            return
        if not self.using_faiss:
            # nothing to build for keyword fallback
            return
        # build embeddings + faiss index
        from sentence_transformers import SentenceTransformer
        import numpy as np
        import faiss

        self.emb_model = SentenceTransformer(self.model_name)
        corpus = [d["text"] for d in self.docs]
        embeddings = self.emb_model.encode(corpus, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        dim = int(embeddings.shape[1])
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype("float32"))

    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        # If faiss available, use vector search; otherwise fallback to simple substring match
        if self.using_faiss and self.index is not None:
            q_emb = self.emb_model.encode([query], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
            D, I = self.index.search(q_emb, k)
            results = []
            for score, idx in zip(D[0], I[0]):
                if idx == -1:
                    continue
                results.append({"text": self.docs[int(idx)]["text"], "metadata": self.docs[int(idx)].get("metadata", {}), "score": float(score)})
            return results
        # fallback: keyword match scoring
        q = query.lower().split()
        scored = []
        for d in self.docs:
            text = d["text"].lower()
            score = sum(1 for w in q if w and w in text)
            if score > 0:
                scored.append((score, d))
        scored.sort(key=lambda x: -x[0])
        return [{"text": d["text"], "metadata": d.get("metadata", {}), "score": float(s)} for s, d in scored[:k]]

# ---------- Gemini client with retries and safe parsing ----------
async def gemini_generate_async(
    api_key: str,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 1200,
    retries: int = 2,
) -> str:
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}

    concise_guardrail = (
        "Be concise and fit within token limits. Use explicit sections: Root Cause, Evidence, Prediction, Immediate Action, Long-term Mitigation."
    )

    for attempt in range(retries + 1):
        effective_max = max(256, int(max_output_tokens * (0.7 ** attempt)))
        parts = []
        if system_prompt:
            parts.append({"text": f"[SYSTEM]\n{system_prompt}\n\n{concise_guardrail}".strip()})
        else:
            parts.append({"text": concise_guardrail})
        parts.append({"text": user_prompt})
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": effective_max,
                "responseMimeType": "text/plain",
            },
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(API_URL, headers=headers, params=params, json=payload, timeout=120) as resp:
                    text_body = await resp.text()
                    if resp.status != 200:
                        # retry on transient errors
                        if resp.status in (429, 500, 502, 503, 504) and attempt < retries:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        raise RuntimeError(f"Gemini API error {resp.status}: {text_body}")
                    data = json.loads(text_body)
                    # try to extract generated text safely
                    try:
                        candidate = data.get("candidates", [None])[0]
                        if candidate and isinstance(candidate, dict):
                            content = candidate.get("content", {})
                            parts = content.get("parts", [])
                            if parts and isinstance(parts, list) and "text" in parts[0]:
                                txt = parts[0]["text"]
                                if txt and txt.strip():
                                    return txt
                    except Exception:
                        pass
                    # If we get here, maybe truncated/finishReason; retry a smaller output token budget
                    finish_reason = None
                    try:
                        finish_reason = data.get("candidates", [{}])[0].get("finishReason")
                    except Exception:
                        finish_reason = None
                    if attempt < retries and finish_reason == "MAX_TOKENS":
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    # fallback: return pretty JSON so user can inspect
                    return json.dumps(data, indent=2)
        except aiohttp.ClientError as e:
            if attempt < retries:
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            return f"Network error contacting Gemini API: {e}"
    return "Failed to get response from Gemini API."

def gemini_generate(*args, **kwargs) -> str:
    return try_run(gemini_generate_async(*args, **kwargs))

# ---------- Agentic pipeline for predictive hardware failure ----------
def agentic_predictive_pipeline(api_key: str, incident: str, rag: SimpleRAG, temperature: float = 0.2) -> str:
    # 1) generate short hypotheses
    hypo_prompt = f"""You are a senior data center reliability engineer. Provide 3 concise hypotheses (1 line each) about what hardware failures or precursors might explain the following telemetry/logs. Be brief.

Incident/Telemetry:
{incident}
"""
    hypotheses = gemini_generate(api_key=api_key, user_prompt=hypo_prompt, system_prompt="Generate 3 short hypotheses.", temperature=temperature, max_output_tokens=200)

    # 2) generate up to 5 focused queries for RAG retrieval
    qry_prompt = f"""From these hypotheses, produce up to 5 short keyword queries suitable for searching logs or KB for evidence. Output one query per line.

Hypotheses:
{hypotheses}
"""
    queries_text = gemini_generate(api_key=api_key, user_prompt=qry_prompt, system_prompt="Output plain queries, one per line.", temperature=0.1, max_output_tokens=200)
    queries = [q.strip(" -•\n\r\t") for q in queries_text.splitlines() if q.strip()][:5]

    # 3) retrieve context from RAG
    retrieved = []
    for q in queries:
        hits = rag.similarity_search(q, k=3)
        for h in hits:
            retrieved.append(h["text"])
    # dedupe and trim
    seen = set()
    ctx_parts = []
    for t in retrieved:
        if t and t not in seen:
            seen.add(t)
            ctx_parts.append(t)
    context_text = safe_trim("\n---\n".join(ctx_parts[:10]), 4000)

    # 4) final predictive analysis with solution
    final_prompt = f"""
You are an expert SRE specializing in predictive hardware failure analysis.

Incident/Telemetry:
{incident}

Hypotheses:
{hypotheses}

Retrieved Evidence (from KB & logs):
{context_text if context_text else '[no retrieved context]'}

Produce a concise predictive hardware failure analysis with exactly these sections:
1) Root Cause (likely hardware issue or precursor)
2) Prediction (probability and timeline for failure if trend continues)
3) Evidence (bullet points linking metrics/logs)
4) Immediate Actions (step-by-step commands/actions to mitigate risk now)
5) Long-term Remediation & Monitoring (3 items)

Be concrete and include commands where applicable (linux, ipmitool, vendor CLI) and keep answer concise.
"""
    return gemini_generate(api_key=api_key, user_prompt=final_prompt, system_prompt="Deliver a sharp predictive RCA with actionable steps.", temperature=temperature, max_output_tokens=900)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Predictive Hardware Failure Analysis", layout="wide")
st.title("🔧 MANISH SINGH - Predictive Hardware Failure Analysis (Gemini + RAG + Agent)")

with st.sidebar:
    st.markdown("## 🔐 API Key")
    manual_key = st.text_input("Enter GEMINI_API_KEY (optional)", type="password", key="api_key_input")
    api_key = manual_key or get_api_key()
    if not api_key:
        st.warning("Set GEMINI_API_KEY in Streamlit secrets or environment to call Gemini.")
    st.markdown("---")
    st.markdown("Upload logs, telemetry dumps, or runbooks (txt/pdf) to build an on-instance KB for RAG.")

uploaded_files = st.file_uploader("Upload files (.txt, .log, .pdf)", accept_multiple_files=True, type=["txt", "log", "pdf"])
kb_texts: List[str] = []
if uploaded_files:
    for uf in uploaded_files:
        try:
            if uf.name.lower().endswith((".txt", ".log")):
                txt = uf.read().decode("utf-8", errors="ignore")
                kb_texts.extend(chunk_text(txt, max_chars=1200))
            elif uf.name.lower().endswith(".pdf"):
                reader = None
                try:
                    reader = PdfReader(io.BytesIO(uf.read()))
                    txt = []
                    for p in reader.pages:
                        txt.append(p.extract_text() or "")
                    txt = "\n".join(txt)
                    kb_texts.extend(chunk_text(txt, max_chars=1200))
                except Exception:
                    # fallback: raw bytes decode
                    try:
                        txt = uf.read().decode("utf-8", errors="ignore")
                        kb_texts.extend(chunk_text(txt, max_chars=1200))
                    except Exception:
                        st.warning(f"Could not parse {uf.name}")
        except Exception as e:
            st.warning(f"Failed to read {uf.name}: {e}")

# instantiate RAG
rag = SimpleRAG()
if kb_texts:
    rag.add_documents(kb_texts, [{"source": "upload"} for _ in kb_texts])
    try:
        rag.build()
        st.success(f"Knowledge base ready with ~{len(kb_texts)} chunks.")
    except Exception as e:
        st.warning(f"RAG build failed (faiss/sentence-transformers may be missing): {e}. Falling back to keyword retrieval.")

incident_input = st.text_area("Paste telemetry/logs/incident description here (or a short summary):", height=240, key="incident_input")
if st.button("Run Predictive Analysis", key="run_analysis"):
    if not api_key:
        st.error("Gemini API key required.")
    elif not incident_input.strip():
        st.warning("Please paste the telemetry or incident text to analyze.")
    else:
        with st.spinner("Running agentic predictive analysis..."):
            try:
                report = agentic_predictive_pipeline(api_key=api_key, incident=incident_input.strip(), rag=rag, temperature=0.2)
            except Exception as e:
                report = f"Error running analysis: {e}"
        st.subheader("📄 Predictive Hardware Failure Analysis")
        st.markdown(report)

if st.button("Load example incident"):
    example = (
        "Node: server-23\n"
        "Metric: disk_read_latency_ms=45 (baseline 5ms), disk_write_latency_ms=80 (baseline 7ms)\n"
        "SMART: reallocated_sector_count=120, current_pending_sector=4\n"
        "Event: repeated soft ECC errors reported on channel A\n"
        "Recent change: firmware update to storage controller 2 days ago\n"
    )
    st.session_state["incident_input"] = example
    st.experimental_rerun()
