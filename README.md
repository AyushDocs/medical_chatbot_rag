# Medical Chatbot with RAG

A Retrieval-Augmented Generation (RAG) chatbot that answers medical questions using LangChain + FAISS.

**[Live Demo](https://huggingface.co/spaces/AyushDocs/medical_chatbot_rag)**

## How It Works

1. **PDF → Chunks** — Medical PDF is split into ~500-char chunks
2. **Chunks → Embeddings** — Each chunk is embedded with `all-MiniLM-L6-v2`
3. **Embeddings → FAISS Index** — Stored for fast similarity search
4. **Query → Retrieve → Generate** — User question retrieves top-4 chunks, LLM generates answer

## Local Setup

```bash
git clone https://github.com/AyushDocs/medical_chatbot_rag.git
cd medical_chatbot_rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Build FAISS index (place PDF in data/ first)
python store_index.py

# Run Flask app
python app.py
```

## Models

| Component | Model | Size |
|-----------|-------|------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 80MB |
| LLM | `google/flan-t5-base` | ~250M params |
