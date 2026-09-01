import os
import json
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
from tqdm import tqdm
from openai import OpenAI

# --- Config ---
FAQ_DIR = os.getenv("FAQ_DIR", str(Path(__file__).parent / "faqs"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-ada-002")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "4"))

# Initialize the OpenAI client (fail fast if key missing)
_API_KEY = os.getenv("OPENAI_API_KEY")
if not _API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")
client = OpenAI(api_key=_API_KEY)

# Globals (preloaded at import)
_CHUNKS: List[str] = []
_SOURCES: List[str] = []
_CHUNK_EMBEDS: np.ndarray | None = None  # shape: (N, d)

# ---------------- Core utilities ----------------
def _chunk_text(text: str, size: int = CHUNK_SIZE) -> List[str]:
    """TODO: Split text into fixed-size chunks and return the list of chunks."""
    raise NotImplementedError

def _load_and_chunk_faqs(faq_dir: str) -> Tuple[List[str], List[str]]:
    """TODO: Load *.md files, chunk each, and return (chunks, matching_source_filenames)."""
    raise NotImplementedError

def _embed_texts(texts: List[str]) -> np.ndarray:
    """TODO: Create embeddings for texts and return a (N, d) float32 numpy array."""
    raise NotImplementedError

def _embed_query(q: str) -> np.ndarray:
    """TODO: Create an embedding for the query and return a (d,) float32 vector."""
    raise NotImplementedError

def _generate_answer(context: str, question: str) -> str:
    """TODO: Call the chat model to answer using only context and cite filenames."""
    raise NotImplementedError

# ---------------- Public API ----------------
def ask_faq_core(question: str, top_k: int = TOP_K_DEFAULT) -> Dict[str, object]:
    q = (question or "").strip()
    if not q:
        raise ValueError("question is required")
    if top_k <= 0:
        top_k = TOP_K_DEFAULT

    # If not yet implemented, return a safe placeholder so wrappers run.
    if _CHUNK_EMBEDS is None or len(_CHUNKS) == 0:
        return {
            "answer": "Placeholder: implement retrieval + grounded generation with citations.",
            "sources": ["faq_auth.md", "faq_employee.md"],
        }

    q_emb = _embed_query(q)

    sims = _CHUNK_EMBEDS @ q_emb  # cosine if rows are normalized
    top_idx = np.argsort(sims)[-top_k:][::-1]
    top_files = [_SOURCES[i] for i in top_idx]
    context_parts = [f"From {_SOURCES[i]}:\n{_CHUNKS[i]}" for i in top_idx]
    context = "\n\n".join(context_parts)

    answer = _generate_answer(context, q)
    distinct_sources = sorted(list({f for f in top_files}))
    sources_out = distinct_sources[:2] if len(distinct_sources) >= 2 else distinct_sources
    return {"answer": answer, "sources": sources_out}

# ---------------- Module preload ----------------
def _preload() -> None:
    """TODO: Load and chunk FAQs, compute embeddings, L2-normalize rows, assign globals."""
    raise NotImplementedError

# Run preload at import time (enable after implementation)
# _preload()

# ---------------- Optional CLI runner ----------------
def main_cli():
    q = input("Enter your question: ")
    print(json.dumps(ask_faq_core(q), indent=2))

if __name__ == "__main__":
    main_cli()
