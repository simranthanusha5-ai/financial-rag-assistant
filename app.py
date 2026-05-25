import streamlit as st

from pdf_extractor import extract_text
from chunking import chunk_text
from embeddings import create_embeddings
from vector_store import store_embeddings
from retriever import retrieve
from llm import ask_llm

st.set_page_config(
    page_title="AI Document RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

.main-title {
    font-size: 48px;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #22d3ee, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sub-title {
    text-align: center;
    font-size: 18px;
    color: #cbd5e1;
    margin-bottom: 30px;
}

.card {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 16px 45px rgba(0,0,0,0.25);
}

.answer-card {
    background: rgba(15, 23, 42, 0.95);
    border-left: 5px solid #22d3ee;
    border-radius: 16px;
    padding: 22px;
    font-size: 16px;
    line-height: 1.7;
}

.metric-card {
    background: rgba(30, 41, 59, 0.8);
    border-radius: 16px;
    padding: 18px;
    text-align: center;
    border: 1px solid rgba(148, 163, 184, 0.15);
}

.metric-num {
    font-size: 28px;
    font-weight: 800;
    color: #a78bfa;
}

.stTextInput input {
    border-radius: 14px;
}

.stFileUploader {
    background: rgba(15, 23, 42, 0.6);
    border-radius: 16px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🤖 AI Document RAG Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Upload a PDF and ask questions using RAG, FAISS, and Groq LLM</div>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.title("⚡ RAG Assistant")
    st.write("Upload a PDF and ask questions.")
    st.markdown("---")
    st.write("**Tech Stack**")
    st.write("• Streamlit")
    st.write("• PyMuPDF")
    st.write("• Sentence Transformers")
    st.write("• FAISS")
    st.write("• Groq LLM")
    st.markdown("---")
    st.info("Best for text-based PDFs. Scanned PDFs need OCR.")

left, right = st.columns([1, 1.5], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📄 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )

    st.markdown("### Try asking")
    st.write("• What is this document about?")
    st.write("• Summarize the key points")
    st.write("• What technical skills are mentioned?")
    st.write("• What were the net sales?")
    st.write("• What are the main projects?")

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💬 Ask Your Question")

    if uploaded_file is None:
        st.info("Upload a PDF to begin.")
    else:
        with open("uploaded.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("PDF uploaded successfully")

        with st.spinner("Processing document..."):
            text = extract_text("uploaded.pdf")

            if not text.strip():
                st.error("No readable text found. This may be a scanned/image PDF.")
                st.stop()

            chunks = chunk_text(text)

            if len(chunks) == 0:
                st.error("No chunks created. Please try another PDF.")
                st.stop()

            embeddings = create_embeddings(chunks)
            index = store_embeddings(embeddings)

        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-num">{len(text):,}</div>Characters</div>',
                unsafe_allow_html=True
            )

        with m2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-num">{len(chunks)}</div>Chunks</div>',
                unsafe_allow_html=True
            )

        with m3:
            st.markdown(
                '<div class="metric-card"><div class="metric-num">FAISS</div>Vector DB</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        question = st.text_input(
            "Ask something from the PDF",
            placeholder="Example: Summarize this document"
        )

        if question:
            with st.spinner("Generating answer..."):
                retrieved_chunks = retrieve(question, index, chunks, top_k=8)
                context = "\n".join(retrieved_chunks)

                if not context.strip():
                    st.error("No relevant context found. Try another question.")
                    st.stop()

                answer = ask_llm(context, question)

            st.markdown("### ✨ Answer")
            st.markdown(
                f'<div class="answer-card">{answer}</div>',
                unsafe_allow_html=True
            )

            with st.expander("🔍 Retrieved Context"):
                st.write(context)

            with st.expander("📃 Extracted Text Preview"):
                st.write(text[:1500])

    st.markdown("</div>", unsafe_allow_html=True)