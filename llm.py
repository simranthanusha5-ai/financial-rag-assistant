from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_llm(context, question):
    prompt = f"""
You are DocuMind AI, a helpful document question-answering assistant.

Use the provided context to answer the user's question.

Rules:
- Answer only using the context.
- If the user asks for a summary, summarize the context clearly.
- If the document is notes, explain concepts simply.
- If the document is a resume, answer about skills, education, projects, and experience.
- If the document is financial, answer with exact values when available.
- If the answer is not in the context, say "Not Found".
- Keep the answer clear, useful, and direct.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content