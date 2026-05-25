import streamlit as st

from pdf_extractor import extract_text
from chunking import chunk_text
from embeddings import create_embeddings
from vector_store import store_embeddings
from retriever import retrieve
from llm import ask_llm

st.set_page_config(
    page_title="Financial RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

.main-card {
    background: rgba(255, 255, 255, 0.08);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    animation: fadeIn 1s ease-in-out;
}

.title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #38bdf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cbd5e1;
    margin-bottom: 30px;
}

.answer-box {
    background: rgba(14, 165, 233, 0.15);
    padding: 20px;
    border-radius: 16px;
    border-left: 5px solid #38bdf8;
    animation: slideUp 0.7s ease;
}

@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}

@keyframes slideUp {
    from {transform: translateY(20px); opacity: 0;}
    to {transform: translateY(0); opacity: 1;}
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🤖 AI Document RAG Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a PDF and ask intelligent questions using local AI + RAG</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="main-card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📄 Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ PDF uploaded successfully")

    with st.spinner("Reading and indexing document..."):
        text = extract_text("uploaded.pdf")
        chunks = chunk_text(text)
        embeddings = create_embeddings(chunks)
        index = store_embeddings(embeddings)

    st.info(f"Document processed successfully. Total chunks created: {len(chunks)}")

    question = st.text_input("💬 Ask a question from your document")

    if question:
        with st.spinner("Thinking with local Qwen model..."):
            retrieved_chunks = retrieve(question, index, chunks)
            context = "\n".join(retrieved_chunks)
            answer = ask_llm(context, question)

        st.markdown("### ✨ Answer")
        st.markdown(
            f'<div class="answer-box">{answer}</div>',
            unsafe_allow_html=True
        )

        with st.expander("🔍 Retrieved Context"):
            st.write(context)

st.markdown('</div>', unsafe_allow_html=True)