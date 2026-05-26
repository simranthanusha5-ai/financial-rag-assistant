import streamlit as st

from pdf_extractor import extract_text
from chunking import chunk_text
from embeddings import create_embeddings
from vector_store import store_embeddings
from retriever import retrieve
from llm import ask_llm

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📘",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #0b0b0b;
    color: #f5f5f5;
}

.block-container {
    padding-top: 2rem;
    max-width: 1150px;
}

[data-testid="stSidebar"] {
    background: #111111;
    border-right: 1px solid #2a2a2a;
}

.title {
    font-size: 48px;
    font-weight: 900;
    text-align: center;
    color: #f5c542;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #cfcfcf;
    font-size: 18px;
    margin-bottom: 35px;
}

.card {
    background: #151515;
    border: 1px solid #2d2d2d;
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 12px 35px rgba(0,0,0,0.35);
}

.answer-box {
    background: #101010;
    border-left: 5px solid #f5c542;
    border-radius: 14px;
    padding: 22px;
    line-height: 1.7;
    font-size: 16px;
}

.metric-box {
    background: #1c1c1c;
    border: 1px solid #333;
    border-radius: 14px;
    padding: 18px;
    text-align: center;
}

.metric-value {
    color: #f5c542;
    font-size: 26px;
    font-weight: 800;
}

.example-box {
    background: #1b1b1b;
    border: 1px solid #303030;
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 8px;
}

.stTextInput input {
    background: #111 !important;
    color: white !important;
    border: 1px solid #444 !important;
    border-radius: 12px !important;
}

.stFileUploader {
    background: #111;
    border: 1px solid #333;
    border-radius: 14px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("📘 DocuMind AI")
    st.write("Ask questions from any text-based PDF.")
    st.divider()
    st.write("**Works with:**")
    st.write("• Notes")
    st.write("• Resumes")
    st.write("• Financial reports")
    st.write("• Research papers")
    st.write("• Articles")
    st.divider()
    st.write("**Tech Stack**")
    st.write("• Streamlit")
    st.write("• PyMuPDF")
    st.write("• Sentence Transformers")
    st.write("• FAISS")
    st.write("• Groq LLM")
    st.divider()
    st.warning("Scanned/image PDFs need OCR.")

st.markdown('<div class="title">📘 DocuMind AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload any text-based PDF and ask questions using RAG + FAISS + LLM</div>',
    unsafe_allow_html=True
)

left, right = st.columns([1, 1.6], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📄 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    st.markdown("### Try asking")

    examples = [
        "What is this document about?",
        "Summarize this in simple English",
        "What are the key points?",
        "Explain the important concepts",
        "List the main topics covered",
        "What technical skills are mentioned?",
        "What were the net sales?"
    ]

    for ex in examples:
        st.markdown(f'<div class="example-box">{ex}</div>', unsafe_allow_html=True)

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

        with st.spinner("Reading and indexing document..."):
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
                f'<div class="metric-box"><div class="metric-value">{len(text):,}</div>Characters</div>',
                unsafe_allow_html=True
            )

        with m2:
            st.markdown(
                f'<div class="metric-box"><div class="metric-value">{len(chunks)}</div>Chunks</div>',
                unsafe_allow_html=True
            )

        with m3:
            st.markdown(
                '<div class="metric-box"><div class="metric-value">FAISS</div>Vector DB</div>',
                unsafe_allow_html=True
            )

        st.divider()

        question = st.text_input(
            "Ask something from the PDF",
            placeholder="Example: Summarize this chapter"
        )

        if question:
            with st.spinner("Generating answer..."):
                retrieved_chunks = retrieve(question, index, chunks, top_k=10)
                context = "\n".join(retrieved_chunks)

                if not context.strip():
                    st.error("No relevant context found. Try rephrasing the question.")
                    st.stop()

                answer = ask_llm(context, question)

            st.markdown("### ✨ Answer")
            st.markdown(
                f'<div class="answer-box">{answer}</div>',
                unsafe_allow_html=True
            )

            with st.expander("🔍 Retrieved Context"):
                st.write(context)

            with st.expander("📃 Extracted Text Preview"):
                st.write(text[:2000])

    st.markdown("</div>", unsafe_allow_html=True)