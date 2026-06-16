import os
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

DATA_PATH = "data/medical book.pdf"
FAISS_PATH = "models"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_vector_store():
    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    print(f"Using device: {DEVICE}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": DEVICE},
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    os.makedirs(os.path.dirname(FAISS_PATH), exist_ok=True)
    vector_store.save_local(FAISS_PATH)
    print(f"Saved FAISS index to {FAISS_PATH}")


if __name__ == "__main__":
    create_vector_store()
