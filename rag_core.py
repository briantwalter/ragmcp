"""Persistent FAQ RAG core used by both the interactive CLI and MCP server.

Deployment-specific settings come from environment variables so the same module
can run from a terminal, an MCP client, or a container. Chroma is opened lazily
and reused for the lifetime of the process.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import chromadb
from chromadb.api.models.Collection import Collection
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
FAQ_DIR = Path(os.getenv("FAQ_DIR", BASE_DIR / "faqs"))
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", BASE_DIR / ".chroma"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "faq_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "4"))

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set")
if CHUNK_SIZE <= 0:
    raise RuntimeError("CHUNK_SIZE must be greater than zero")
if not 1 <= TOP_K_DEFAULT <= 10:
    raise RuntimeError("TOP_K_DEFAULT must be between 1 and 10")

client = OpenAI()

_chroma_client: Any = None
_collection: Collection | None = None


def _split_long_text(text: str, size: int) -> List[str]:
    """Split one paragraph near word boundaries, falling back to hard limits."""
    chunks: List[str] = []
    current = ""
    for word in text.split():
        if len(word) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(word[i : i + size] for i in range(0, len(word), size))
        elif not current:
            current = word
        elif len(current) + 1 + len(word) <= size:
            current = f"{current} {word}"
        else:
            chunks.append(current)
            current = word
    if current:
        chunks.append(current)
    return chunks


def _chunk_text(text: str, size: int = CHUNK_SIZE) -> List[str]:
    """Create approximately size-character chunks while preserving paragraphs."""
    if size <= 0:
        raise ValueError("chunk size must be greater than zero")

    paragraphs = [" ".join(part.split()) for part in text.split("\n\n")]
    units: List[str] = []
    for paragraph in filter(None, paragraphs):
        units.extend(_split_long_text(paragraph, size))

    chunks: List[str] = []
    current = ""
    for unit in units:
        separator = "\n\n" if current else ""
        if current and len(current) + len(separator) + len(unit) > size:
            chunks.append(current)
            current = unit
        else:
            current = f"{current}{separator}{unit}"
    if current:
        chunks.append(current)
    return chunks


def _load_and_chunk_faqs(faq_dir: Path = FAQ_DIR) -> List[Tuple[str, int, str]]:
    """Return (source filename, per-file chunk index, chunk text) records."""
    if not faq_dir.is_dir():
        raise RuntimeError(f"FAQ directory does not exist: {faq_dir}")

    records: List[Tuple[str, int, str]] = []
    for path in sorted(faq_dir.glob("*.md")):
        for chunk_index, chunk in enumerate(_chunk_text(path.read_text(encoding="utf-8"))):
            records.append((path.name, chunk_index, chunk))
    if not records:
        raise RuntimeError(f"No FAQ content found in {faq_dir}")
    return records


def _chunk_id(source: str, chunk_index: int, content: str) -> str:
    """Build a content-addressed ID used to detect additions and edits."""
    payload = f"{source}\0{chunk_index}\0{content}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _embed_texts(texts: Sequence[str]) -> List[List[float]]:
    """Embed texts and return vectors in the same order as the input sequence.

    Response items are sorted by their API index so input ordering is preserved
    even if an API implementation returns items out of order.
    """
    if not texts:
        return []
    response = client.embeddings.create(model=EMBED_MODEL, input=list(texts))

    ordered = sorted(response.data, key=lambda item: item.index)
    if len(ordered) != len(texts):
        raise RuntimeError("Embedding API returned an unexpected number of vectors")
    return [item.embedding for item in ordered]


def _embed_query(question: str) -> List[float]:
    """Embed one user question for comparison with stored chunk vectors."""
    return _embed_texts([question])[0]


def _collection_metadata() -> Dict[str, str | int]:
    """Describe the vector-space settings that must match persisted data."""
    return {
        "hnsw:space": "cosine",
        "embed_model": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
    }


def _get_collection() -> Collection:
    """Open or create a Chroma collection compatible with current settings.

    A collection created with another embedding model, chunk size, or distance
    metric is deleted because those vectors cannot safely share one index.
    Chroma releases expose different exception classes for missing collections,
    so their stable error messages are used to distinguish that expected case.
    """
    global _chroma_client, _collection
    if _collection is not None:
        return _collection

    _chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        existing = _chroma_client.get_collection(CHROMA_COLLECTION)
        metadata = existing.metadata or {}

        if (
            metadata.get("embed_model") != EMBED_MODEL
            or metadata.get("chunk_size") != CHUNK_SIZE
            or metadata.get("hnsw:space") != "cosine"
        ):
            _chroma_client.delete_collection(CHROMA_COLLECTION)
    except Exception as exc:
        message = str(exc).lower()
        if "does not exist" not in message and "not found" not in message:
            raise

    _collection = _chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata=_collection_metadata(),
    )
    return _collection


def _sync_index() -> Tuple[Collection, Dict[str, int]]:
    """Synchronize FAQs and return the collection plus update statistics.

    Stale records are removed when sources are deleted, renamed, or edited.
    Only new content IDs are embedded; matching IDs reuse persisted vectors.
    """
    collection = _get_collection()
    records = _load_and_chunk_faqs()
    current = {
        _chunk_id(source, chunk_index, content): (source, chunk_index, content)
        for source, chunk_index, content in records
    }
    stored_ids = set(collection.get(include=[])["ids"])
    current_ids = set(current)

    stale_ids = sorted(stored_ids - current_ids)
    if stale_ids:
        collection.delete(ids=stale_ids)

    new_ids = sorted(current_ids - stored_ids)
    if new_ids:
        new_records = [current[chunk_id] for chunk_id in new_ids]
        documents = [record[2] for record in new_records]
        collection.upsert(
            ids=new_ids,
            documents=documents,
            embeddings=_embed_texts(documents),
            metadatas=[
                {"source": source, "chunk_index": chunk_index}
                for source, chunk_index, _ in new_records
            ],
        )
    stats = {
        "source_files": len({source for source, _, _ in records}),
        "chunks_total": len(current_ids),
        "chunks_added": len(new_ids),
        "chunks_deleted": len(stale_ids),
        "chunks_unchanged": len(current_ids & stored_ids),
    }
    return collection, stats


def reindex() -> Dict[str, int]:
    """Synchronize the FAQ directory with Chroma and return update statistics."""
    _, stats = _sync_index()
    return stats


def _generate_answer(context: str, question: str, sources: Sequence[str]) -> str:
    """Generate a context-grounded answer with inline source citations."""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only from the supplied FAQ context. If the context does not contain "
                    "the answer, say that the FAQs do not provide it. Cite supporting filenames "
                    "inline in square brackets. When the evidence supports it and at least two "
                    "distinct filenames are supplied, cite at least two. Never invent a filename."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Available source filenames: {', '.join(sources)}\n\n"
                    f"FAQ context:\n{context}\n\nQuestion: {question}"
                ),
            },
        ],
    )
    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("LLM returned an empty answer")
    return answer.strip()


def ask_faq_core(question: str, top_k: int = TOP_K_DEFAULT) -> Dict[str, object]:
    """Retrieve FAQ chunks and return a grounded answer with source filenames.

    Chroma ranks stored vectors with cosine distance. Source names retain
    retrieval order while duplicates are removed. Each context chunk is labelled
    with its filename so the LLM has an explicit inline citation target.
    """
    q = (question or "").strip()
    if not q:
        raise ValueError("question is required")
    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10")

    collection = _get_collection()
    chunk_count = collection.count()
    if chunk_count == 0:
        raise RuntimeError(
            "FAQ index is empty; run python rag_core.py --reindex before querying"
        )
    result_count = min(top_k, chunk_count)

    results = collection.query(
        query_embeddings=[_embed_query(q)],
        n_results=result_count,
        include=["documents", "metadatas", "distances"],
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    sources = list(dict.fromkeys(str(metadata["source"]) for metadata in metadatas))

    context = "\n\n".join(
        f"From {metadata['source']}:\n{document}"
        for document, metadata in zip(documents, metadatas)
    )
    return {
        "answer": _generate_answer(context, q, sources),
        "sources": sources,
    }


def main_cli(argv: Sequence[str] | None = None) -> None:
    """Run either an explicit reindex or an interactive FAQ query."""
    parser = argparse.ArgumentParser(description="Query or reindex the FAQ corpus.")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="synchronize FAQ files with Chroma, print statistics, and exit",
    )
    args = parser.parse_args(argv)
    if args.reindex:
        print(json.dumps(reindex(), indent=2))
        return

    question = input("Enter your question: ")
    print(json.dumps(ask_faq_core(question), indent=2))


if __name__ == "__main__":
    main_cli()
