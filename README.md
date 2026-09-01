# FAQ RAG + MCP Tool (Starter Skeleton)

This is a minimal starting point for the MCP option.

## Contents
- `rag_core.py` — RAG core
- `mcp_server.py` — MCP server exposing `ask_faq`
- `faqs/` — tiny sample corpus
- `requirements.txt`

## Quick Start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY=sk-...

# Optional model overrides
# export EMBED_MODEL=text-embedding-ada-002
# export LLM_MODEL=gpt-3.5-turbo

# Run a quick CLI smoke test
python rag_core.py

# Configure your MCP client to spawn the server
# command: python
# args: [/absolute/path/to/mcp_server.py]
# env: { OPENAI_API_KEY: "sk-..." }
```