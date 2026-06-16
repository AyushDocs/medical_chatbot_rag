# Medical Chatbot with RAG

A Retrieval-Augmented Generation (RAG) chatbot that answers medical questions using LangChain + FAISS.

## How It Works

1. **PDF → Chunks** — Medical PDF is split into ~500-char chunks
2. **Chunks → Embeddings** — Each chunk is embedded with `all-MiniLM-L6-v2`
3. **Embeddings → FAISS Index** — Stored for fast similarity search
4. **Query → Retrieve → Generate** — User question retrieves top-4 chunks, LLM generates answer

## Quick Start

### Option 1: Google Colab (Recommended)
Open `medical_chatbot_colab.ipynb` in Colab — runs the full pipeline on GPU.

### Option 2: Local
```bash
# Clone
git clone https://github.com/AyushDocs/medical_chatbot_rag.git
cd medical_chatbot_rag

# Install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Build FAISS index (run once)
# Place your PDF in data/medical book.pdf first
python store_index.py

# Run the app
python app.py
```
Open http://localhost:8000

## Project Structure

```
├── app.py                  # Flask web server
├── store_index.py          # Builds FAISS index from PDF
├── src/
│   ├── helper.py           # QA chain (retrieval + LLM)
│   └── prompt.py           # Prompt template
├── templates/
│   └── chat.html           # Frontend UI
├── models/                 # FAISS index (generated, not in repo)
├── data/                   # PDF files (generated, not in repo)
└── medical_chatbot_colab.ipynb  # Colab notebook
```

## Models

| Component | Model | Size |
|-----------|-------|------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 80MB |
| LLM | `google/flan-t5-base` | ~250M params |

> **Note:** `flan-t5-base` is small and may produce low-quality answers. Consider upgrading to a larger model or using an API (OpenAI/Gemini) for production use.

## Requirements

- Python 3.10+
- See `requirements.txt` for dependencies
