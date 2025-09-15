#!/usr/bin/env python3
"""
Llama3 API (Redis + LangChain RAG + SSE)
=======================================

Purpose
-------
Text-only Llama 3 chat service with:
- Rolling session memory in Redis (per session_id)
- Optional RAG via FAISS (LangChain)
- SSE streaming at /stream that emits *accumulated* frames: {"acc": "..."}.

Who uses this
------------
- Orchestrator (control plane): preview/assistant text during /chat with
  session_id (thread_id) so “also add habitat” works without repeating context.
- UI: optional direct /stream for live typing; optional /reset to clear memory.

What it returns
--------------
- POST /chat   -> {"reply": "..."}           # sync reply
- POST /stream -> SSE frames {"acc":"..."}   # accumulated full text each frame
- POST /reset  -> {"ok": true, "session_cleared": "<id>"}
- POST /rag/reload -> {"ok": true|false}
- GET  /health -> model + env diagnostics

Design notes
------------
- Mirrors the Qwen server’s contract so the UI/orchestrator can reuse the same
  code paths and streaming logic. (Frames are FULL accumulations: {"acc": ...})
- Memory schema aligns with your Qwen2-VL service for parity.  # see: qwen2_vl_api_langchain_rag_redis_sse.py
- Generation uses Hugging Face Transformers AutoModelForCausalLM + TextIteratorStreamer.
"""

import os, io, json, time, shutil, threading, re, base64
from typing import List, Dict, Any, Optional, Tuple
import logging

import requests
import redis
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    GenerationConfig,
)

# -------- LangChain / FAISS (optional RAG) --------
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# =============================================================================
# Paths / environment
# =============================================================================
logger = logging.getLogger("llama3_api")
logging.basicConfig(level=logging.INFO)

BASE_DIR  = os.path.expanduser(os.environ.get("BASE_DIR", "~/project"))
MEDIA_DIR = os.path.join(BASE_DIR, "media")
LOGS_DIR  = os.path.join(BASE_DIR, "logs")

FAISS_DIR   = os.path.expanduser(os.environ.get("FAISS_DIR", os.path.join(BASE_DIR, "faiss_index")))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RAG_TOP_K   = int(os.environ.get("RAG_TOP_K", "3"))

# Honor HF caches (similar to your serve_model.py)  # serve_model.py
for k in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
    v = os.getenv(k)
    if v:
        os.environ[k] = os.path.abspath(os.path.expanduser(v))  # normalize  # serve_model.py

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(MEDIA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# =============================================================================
# Redis session memory
# =============================================================================
REDIS_URL   = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
REDIS_PREF  = os.environ.get("REDIS_PREFIX", "llama3")
SESSION_TTL = int(os.environ.get("SESSION_TTL", "604800"))  # 7 days

rds = redis.Redis.from_url(REDIS_URL, decode_responses=True)
def _sk(sid: str) -> str: return f"{REDIS_PREF}:session:{sid}"

def load_session(sid: str) -> Dict[str, Any]:
    """Load a conversation from Redis; create a blank shell on first use."""
    raw = rds.get(_sk(sid))
    if raw:
        return json.loads(raw)
    return {"system": "You are a helpful assistant.", "messages": []}

def save_session(sid: str, data: Dict[str, Any]):
    """Persist conversation to Redis with TTL (rolling)."""
    rds.setex(_sk(sid), SESSION_TTL, json.dumps(data))

def delete_session(sid: str):
    """Hard reset: drop Redis key and any session media folder."""
    rds.delete(_sk(sid))
    sess_dir = os.path.join(MEDIA_DIR, sid)
    if os.path.isdir(sess_dir):
        shutil.rmtree(sess_dir, ignore_errors=True)

# =============================================================================
# Model (Llama 3)
# =============================================================================
MODEL_REPO = os.getenv("MODEL_REPO", "meta-llama/Meta-Llama-3-8B-Instruct")
HUB_TOKEN  = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

# tokenizer & chat template (aligns with serve_model.py pattern)  # serve_model.py
tok = AutoTokenizer.from_pretrained(MODEL_REPO, token=HUB_TOKEN, use_fast=True)
if not getattr(tok, "chat_template", None):
    tok.chat_template = (
        "<|begin_of_text|>{% for m in messages %}"
        "{% if m['role']=='system' %}{{ '<|start_header_id|>system<|end_header_id|>\n\n' + m['content'] + '<|eot_id|>' }}"
        "{% elif m['role']=='user' %}{{ '<|start_header_id|>user<|end_header_id|>\n\n' + m['content'] + '<|eot_id|>' }}"
        "{% elif m['role']=='assistant' %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + m['content'] + '<|eot_id|>' }}"
        "{% endif %}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    )

model = AutoModelForCausalLM.from_pretrained(
    MODEL_REPO,
    token=HUB_TOKEN,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    attn_implementation=os.getenv("ATTN_IMPL", "sdpa"),
).eval()

if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
stop_ids = [tok.eos_token_id]
try:
    eot_id = tok.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id != -1:
        stop_ids = list(set(stop_ids + [eot_id]))
except Exception:
    pass

def _render_messages_to_inputs(messages: List[Dict[str, str]]):
    """Use chat_template to render an inputs batch suitable for generation."""
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt").to(model.device)

# =============================================================================
# RAG (LangChain/FAISS) -- optional
# =============================================================================
retriever = None
def load_retriever():
    """Try to load a FAISS index from FAISS_DIR; if missing, disable RAG."""
    global retriever
    if os.path.isdir(FAISS_DIR):
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        retriever = FAISS.load_local(
            FAISS_DIR, embeddings, allow_dangerous_deserialization=True
        ).as_retriever(search_kwargs={"k": RAG_TOP_K})
        logger.info("RAG retriever loaded from %s", FAISS_DIR)
    else:
        retriever = None
        logger.info("RAG retriever not found (FAISS_DIR=%s)", FAISS_DIR)
    return retriever

load_retriever()

# =============================================================================
# Anti-repetition (outside code blocks)
# =============================================================================
_WORD_RE  = re.compile(r"\b(\w+)(\s+\1){1,}\b", flags=re.IGNORECASE)
_BIGR_RE  = re.compile(r"\b(\w+\s+\w+)(\s+\1){1,}\b", flags=re.IGNORECASE)
_ENUM_RE  = re.compile(r"\b(\d+\s*[.)])(\s*\1){1,}\b")
_COLON_RE = re.compile(r"\b([A-Za-z]+:\s*)(\1){1,}\b", flags=re.IGNORECASE)
_CAMEL_JOIN = re.compile(r"([a-z])([A-Z])")

def _collapse_outside_code(text: str) -> str:
    parts = text.split("```")
    for i in range(0, len(parts), 2):  # even parts are outside code fences
        seg = parts[i]
        seg = _WORD_RE.sub(r"\1", seg)
        seg = _BIGR_RE.sub(r"\1", seg)
        seg = _ENUM_RE.sub(r"\1", seg)
        seg = _COLON_RE.sub(r"\1", seg)
        seg = _CAMEL_JOIN.sub(r"\1 \2", seg)
        parts[i] = seg
    return "```".join(parts)

def _gen_config(max_new_tokens: int, temperature: float) -> GenerationConfig:
    return GenerationConfig(
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.15,
        max_new_tokens=max_new_tokens,
        eos_token_id=stop_ids,
        pad_token_id=tok.pad_token_id,
    )

# =============================================================================
# History helpers (messages = [{"role": "...", "content": "..."}])
# =============================================================================
def _ensure_system(sess: Dict[str, Any]):
    if not any(m["role"] == "system" for m in sess["messages"]):
        sys_msg = sess.get("system") or "You are a helpful assistant."
        sess["messages"].append({"role": "system", "content": sys_msg})

def _trim_pairs(sess: Dict[str, Any], max_turns: int):
    """Keep up to N user/assistant pairs (plus a system message)."""
    msgs = sess["messages"]
    sys_msgs = [m for m in msgs if m["role"] == "system"][:1]
    convo = [m for m in msgs if m["role"] != "system"]
    pairs, i = [], 0
    while i < len(convo):
        u = convo[i] if convo[i]["role"] == "user" else None
        a = convo[i+1] if i+1 < len(convo) and convo[i+1]["role"] == "assistant" else None
        if u: pairs.append((u, a)); i += 2
        else: i += 1
    kept = pairs[-max_turns:] if max_turns > 0 else []
    flat = sys_msgs[:]
    for u, a in kept:
        flat.append(u);
        if a: flat.append(a)
    sess["messages"] = flat

# =============================================================================
# Schemas
# =============================================================================
class ChatRequest(BaseModel):
    session_id: str = Field(default="default", description="Opaque conversation id (thread_id)")
    prompt: str = Field(..., description="User message")
    use_rag: Optional[bool] = Field(default=False, description="Augment with FAISS context if available")
    max_new_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.4
    top_p: Optional[float] = 0.9   # reserved for future tuning
    max_history_turns: Optional[int] = 8

class ChatResponse(BaseModel):
    reply: str

class ResetRequest(BaseModel):
    session_id: str

# --- OpenAI-compatible schema (from serve_model.py) ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionsRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float | None = 0.0
    max_tokens: int | None = 256


# =============================================================================
# FastAPI app
# =============================================================================
app = FastAPI(title="Llama3 API (Redis+RAG+SSE)", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("UI_ORIGIN", "http://localhost:3000"),
        os.getenv("UI_ORIGIN_ALT", "http://127.0.0.1:3000"),
    ],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "ok": True,
        "model": MODEL_REPO,
        "cuda": torch.cuda.is_available(),
        "gpus": torch.cuda.device_count(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "rag": bool(retriever is not None),
        "redis": True,
        "base_dir": BASE_DIR,
        "faiss_dir": FAISS_DIR,
    }

# --- OpenAI-compatible endpoint (from serve_model.py) ---
@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionsRequest):
    """Stateless, non-streaming endpoint compatible with the OpenAI schema."""
    msgs = [m.dict() for m in req.messages]
    t0 = time.time()

    # Render messages, generate, and decode
    inputs = _render_messages_to_inputs(msgs)
    cfg = _gen_config(
        int(req.max_tokens or 256),
        float(req.temperature or 0.0)
    )
    with torch.inference_mode():
        out = model.generate(**inputs, generation_config=cfg)
    
    gen = out[0, inputs.input_ids.shape[-1]:]
    text = tok.decode(gen, skip_special_tokens=True).strip()
    text = _collapse_outside_code(text)
    dt = time.time() - t0

    # OpenAI-style response
    return {
        "id": f"chatcmpl_{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        "latency_s": round(dt, 3),
    }

@app.post("/rag/reload")
def rag_reload():
    ok = load_retriever() is not None
    return {"ok": ok}

@app.post("/reset")
def reset(req: ResetRequest):
    delete_session(req.session_id)
    return {"ok": True, "session_cleared": req.session_id}

# -----------------------------------------------------------------------------
# /chat  (sync)
# -----------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    sess = load_session(req.session_id)
    _ensure_system(sess)

    # Build the user content with optional RAG
    user_text = req.prompt
    if req.use_rag and retriever is not None:
        docs = retriever.get_relevant_documents(user_text)
        ctx = "\n\n".join(d.page_content for d in docs if d.page_content)
        if ctx.strip():
            user_text = f"{user_text}\n\nCONTEXT:\n{ctx}"

    sess["messages"].append({"role": "user", "content": user_text})
    _trim_pairs(sess, req.max_history_turns or 8)

    # Render for Llama3
    inputs = _render_messages_to_inputs(sess["messages"])
    cfg = _gen_config(int(req.max_new_tokens or 256), float(req.temperature or 0.0))

    with torch.inference_mode():
        out = model.generate(**inputs, generation_config=cfg)

    gen = out[0, inputs.input_ids.shape[-1]:]
    reply = tok.decode(gen, skip_special_tokens=True).strip()
    reply = _collapse_outside_code(reply)

    # Persist assistant turn
    sess["messages"].append({"role": "assistant", "content": reply})
    save_session(req.session_id, sess)

    return ChatResponse(reply=reply)

# -----------------------------------------------------------------------------
# /stream  (SSE with FULL accumulation frames {"acc":"..."})
# -----------------------------------------------------------------------------
@app.post("/stream")
def stream(req: ChatRequest):
    sess = load_session(req.session_id)
    _ensure_system(sess)

    user_text = req.prompt
    if req.use_rag and retriever is not None:
        docs = retriever.get_relevant_documents(user_text)
        ctx = "\n\n".join(d.page_content for d in docs if d.page_content)
        if ctx.strip():
            user_text = f"{user_text}\n\nCONTEXT:\n{ctx}"

    sess["messages"].append({"role": "user", "content": user_text})
    _trim_pairs(sess, req.max_history_turns or 8)

    # Render & streamer
    inputs = _render_messages_to_inputs(sess["messages"])
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    cfg = _gen_config(int(req.max_new_tokens or 256), float(req.temperature or 0.0))

    def _run():
        with torch.inference_mode():
            model.generate(**inputs, generation_config=cfg, streamer=streamer)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    def event_gen():
        acc = ""
        for piece in streamer:
            # accumulate + clean outside code blocks
            acc = _collapse_outside_code(acc + piece)
            yield f"data: {json.dumps({'acc': acc}, ensure_ascii=False)}\n\n"

        # finalize: append assistant turn once
        final = acc.strip()
        sess["messages"].append({"role": "assistant", "content": final})
        save_session(req.session_id, sess)
        yield "data: [DONE]\n\n"

    headers = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive"}
    return StreamingResponse(event_gen(), headers=headers)
