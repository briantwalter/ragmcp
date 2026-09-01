import os
from typing import Dict
from mcp.server.fastmcp import FastMCP
from rag_core import ask_faq_core

# Fail fast on missing key so errors are crisp
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set")

mcp = FastMCP("faq-rag")

@mcp.tool()
def ask_faq(question: str, top_k: int = 4) -> Dict[str, object]:
    """Answer a question from the FAQ corpus and cite at least two files."""
    q = (question or "").strip()
    if not q:
        raise ValueError("`question` is required")
    if top_k <= 0 or top_k > 10:
        top_k = 4
    return ask_faq_core(q, top_k=top_k)

if __name__ == "__main__":
    # STDIO transport (client launches this as a subprocess)
    mcp.run(transport="stdio")
