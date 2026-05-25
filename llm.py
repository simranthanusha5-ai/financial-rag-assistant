from groq import Groq
import os

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def ask_llm(context, question):
    prompt = f"""
You are a document question-answering assistant.

Use ONLY the provided context.
Do not hallucinate.
If answer is missing, say Not Found.

Context:
{context}

Question:
{question}

Answer clearly and accurately.
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