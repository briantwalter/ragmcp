"""STDIO MCP wrapper for the FAQ RAG core.

The API key is validated during startup so configuration errors are immediate.
FastMCP derives the public tool schema from the decorated function. When run as
a script, the MCP client launches this process and communicates over stdin and
stdout.
"""

import os
from typing import Dict
from mcp.server.fastmcp import FastMCP
from rag_core import ask_faq_core


if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set")

mcp = FastMCP("faq-rag")

@mcp.tool()
def ask_faq(question: str, top_k: int = 4) -> Dict[str, object]:
    """Expose FAQ retrieval as an MCP tool returning answer and sources only."""
    q = (question or "").strip()
    if not q:
        raise ValueError("`question` is required")
    if not 1 <= top_k <= 10:
        raise ValueError("`top_k` must be between 1 and 10")
    return ask_faq_core(q, top_k=top_k)

if __name__ == "__main__":
    mcp.run(transport="stdio")
