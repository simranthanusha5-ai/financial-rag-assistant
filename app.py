import streamlit as st

from pdf_extractor import extract_text
from chunking import chunk_text
from embeddings import create_embeddings
from vector_store import store_embeddings
from retriever import retrieve
from llm import ask_llm

st.set_page_config(
    page_title="AI Document RAG Assistant",
    page_icon="⚜️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #000000 0%, #0b0b0b 55%, #141414 100%);
    color: #f8f5e8;
}

.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

[data-testid="stSidebar"] {
    background: #050505;
    border-right: 1px solid rgba(212,175,55,0.35);
}

.main-title {
    text-align: center;
    font-size: 46px;
    font-weight: 900;
    margin-bottom: 5px;
    background: linear-gradient(90deg, #d4af37, #f8e08e, #b8860b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #d6d0b8;
    font-size: 18px;
    margin-bottom: 30px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(18, 18, 18, 0.92);
    border: 1px solid rgba(212,175,55,0.25);
    border-radius: 18px;
    box-shadow: 0 18px 50px rgba(0,0,0,0.45);
}

.stButton button {
    border-radius: 12px;
    background: linear-gradient(90deg, #8b6b00, #d4af37);
    color: black;
    border: none;
    font-weight: 700;
}

.answer-box {
    padding: 22px;
    border-radius: 16px;
    background: rgba(12, 12, 12, 0.98);
    border-left: 5px solid #d4af37;
    border-top: 1px solid rgba(212,175,55,0.25);
    line-height: 1.7;
    font-size: 16px;
    color: #f8f5e8;
}

.metric-box {
    padding: 18px;
    border-radius: 14px;
    background: rgba(22, 22, 22, 0.95);
    border: 1px solid rgba(212,175,55,0.25);
    text-align: center;
}

.metric-value {
    font-size: 26px;
    font-weight: 800;
    color: #f8e08e;
}

.example-box {
    padding: 10px 12px;
    margin-bottom: 8px;
    border-radius: 10px;
    background: rgba(22, 22, 22, 0.9);
    border: 1px solid rgba(212,175,55,0.18);
    color: #f8f5e8;
}

.stTextInput input {
    border-radius: 14px;
    background: #0f0f0f !important;
    color: #f8f5e8 !important;
    border: 1px solid rgba(212,175,55,0.35) !important;
}

.stFileUploader {
    background: rgba(18, 18, 18, 0.9);
    border-radius: 16px;
    padding: 10px;
    border: 1px solid rgba(212,175,55,0.25);
}

div[data-testid="stExpander"] {
    background: rgba(18, 18, 18, 0.9);
    border: 1px solid rgba(212,175,55,0.2);
    border-radius: 14px;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f8e08e;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("⚜️ RAG Assistant")
    st.write("Premium AI PDF Question Answering")
    st.divider()
    st.write("**Tech Stack**")
    st.write("• Streamlit")
    st.write("• PyMuPDF")
    st.write("• Sentence Transformers")
    st.write("• FAISS")
    st.write("• Groq LLM")
    st.divider()
    st.info("Best for text-based PDFs. Scanned PDFs need OCR.")

st.markdown('<div class="main-title">⚜️ AI Document RAG Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a PDF and ask questions using RAG, FAISS, and Groq LLM</div>',
    unsafe_allow_html=True
)

left, right = st.columns([1, 1.6], gap="large")

with left:
    with st.container(border=True):
        st.subheader("📄 Upload PDF")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"]
        )

        st.markdown("### ✨ Try asking")

        examples = [
            "What is this document about?",
            "Summarize the key points",
            "What technical skills are mentioned?",
            "What were the net sales?",
            "What are the main projects?"
        ]

        for ex in examples:
            st.markdown(f'<div class="example-box">{ex}</div>', unsafe_allow_html=True)

with right:
    with st.container(border=True):
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
                placeholder="Example: Summarize this document"
            )

            if question:
                with st.spinner("Generating answer..."):
                    retrieved_chunks = retrieve(question, index, chunks, top_k=8)
                    context = "\\n".join(retrieved_chunks)

                    if not context.strip():
                        st.error("No relevant context found. Try another question.")
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
                    st.write(text[:1500])