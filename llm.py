import ollama

def ask_llm(context, question):
    prompt = f"""
You are a document question-answering assistant.

Use ONLY the provided context to answer the question.
Do not make up information.

Context:
{context}

Question:
{question}

Instructions:
- If the document is financial, answer with financial values clearly.
- If the document is a resume, answer based on skills, education, projects, experience, certifications, and achievements.
- If the answer is not present in the context, say "Not Found".
- Keep the answer clear and structured.
"""

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]