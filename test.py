from pdf_extractor import extract_text
from chunking import chunk_text
from embeddings import create_embeddings
from vector_store import store_embeddings
from retriever import retrieve
from llm import ask_llm

text = extract_text("sample.pdf")
chunks = chunk_text(text)

embeddings = create_embeddings(chunks)
index = store_embeddings(embeddings)

question = "What were Apple's net sales?"

retrieved_chunks = retrieve(question, index, chunks)
context = "\n".join(retrieved_chunks)

answer = ask_llm(context, question)

print("Question:", question)
print("Retrieved Context:", context)
print("Answer:", answer)