# Anciennes versions

Code des versions précédentes, conservé pour référence. Il n'est plus utilisé par l'application.

- `app_ollama.py` : interface Streamlit fonctionnant avec un modèle local via Ollama.
- `agent/` : version RAG avec ChromaDB et embeddings `bge-m3` (Ollama), dont `tools.py`, `provider.py`, `mcp_client.py`, `index_reglement.py` et l'index `.chroma/`.
- `mcp_server/server.py` : ancien serveur MCP basé sur `agent/tools.py`.
- `eval/` : comparaison de modèles locaux (`llama3.2`, `qwen2.5`).

Ces fichiers demandent Ollama en local et `chromadb`, absents de `requirements.txt`.
