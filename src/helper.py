import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from src.prompt import prompt_template
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

FAISS_PATH = "models"
MODEL_NAME = "google/flan-t5-base"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": DEVICE},
    )
    vector_store = FAISS.load_local(
        FAISS_PATH, embeddings, allow_dangerous_deserialization=True
    )
    return vector_store


_tokenizer = None
_model = None


def load_llm():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if _model is None:
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE)
        if DEVICE == "cuda":
            _model = _model.to(DEVICE)
    return _tokenizer, _model


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class QAChain:
    def __init__(self):
        vector_store = load_vector_store()
        self.tokenizer, self.model = load_llm()
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        self.prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

    def invoke(self, input_dict):
        question = input_dict["query"]
        docs = self.retriever.invoke(question)
        context = format_docs(docs)
        prompt_text = self.prompt.format(context=context, question=question)

        inputs = self.tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=512
        )
        if DEVICE == "cuda":
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=256, temperature=0.3, do_sample=True
            )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"result": answer, "source_documents": docs}


def get_qa_chain():
    return QAChain()
