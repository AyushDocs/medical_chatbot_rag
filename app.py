import os
from flask import Flask, render_template, request, jsonify
from src.helper import get_qa_chain

app = Flask(__name__)
qa_chain = None


def init():
    global qa_chain
    faiss_path = os.getenv("FAISS_PATH", "models")
    if not os.path.exists(os.path.join(faiss_path, "index.faiss")):
        return
    qa_chain = get_qa_chain()


@app.route("/")
def index():
    ready = qa_chain is not None
    return render_template("chat.html", ready=ready)


@app.route("/ask", methods=["POST"])
def ask():
    if qa_chain is None:
        return jsonify({"answer": "FAISS index not found. Run store_index.py first."}), 503
    data = request.get_json()
    question = data.get("question", "")
    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    sources = [doc.page_content[:200] for doc in result.get("source_documents", [])]
    return jsonify({"answer": answer, "sources": sources})


if __name__ == "__main__":
    init()
    app.run(debug=False, host="0.0.0.0", port=8000)
