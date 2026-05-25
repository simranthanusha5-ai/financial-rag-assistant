from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_llm(context, question):
    prompt = f"""
You are a smart document question-answering assistant.

Use the provided context to answer the question.

Rules:
- Answer from the context naturally.
- If answer exists partially, infer carefully.
- If person's name exists, return exact name.
- Summarize if user asks summary.
- If numerical value exists, return it clearly.
- If answer is not available anywhere in context, say: Not Found.
- Do NOT refuse unless context is empty.

Context:
{context}

Question:
{question}

Give short, direct answer.
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content