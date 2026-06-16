import os
import gradio as gr
from src.helper import get_qa_chain

qa_chain = None


def init():
    global qa_chain
    if os.path.exists("models/index.faiss"):
        qa_chain = get_qa_chain()
        return "Model loaded! Ask me anything."
    return "No FAISS index found. Please upload a PDF first."


def chat(message, history):
    if qa_chain is None:
        return "Please upload a PDF and build the index first."
    try:
        result = qa_chain.invoke({"query": message})
        answer = result["result"]
        sources = len(result.get("source_documents", []))
        return f"{answer}\n\n*({sources} sources retrieved)*"
    except Exception as e:
        return f"Error: {str(e)}"


demo = gr.ChatInterface(
    fn=chat,
    title="Medical Chatbot",
    description="Ask medical questions. Answers are based on the loaded medical PDF.",
    examples=[
        "What are symptoms of diabetes?",
        "How is hypertension treated?",
        "What causes heart disease?",
    ],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    startup_msg = init()
    print(startup_msg)
    demo.launch(server_name="0.0.0.0", server_port=7860)
