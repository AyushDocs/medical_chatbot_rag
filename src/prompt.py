prompt_template = """
Use the following pieces of context to answer the medical question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer concise and relevant to the clinical context.

Context:
{context}

Question: {question}

Answer:
"""
