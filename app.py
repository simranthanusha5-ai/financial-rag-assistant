import streamlit as st

from pdf_extractor import extract_text
from chunking import chunk_text
from embeddings import create_embeddings
from vector_store import store_embeddings
from retriever import retrieve
from llm import ask_llm

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #050505;
    color: #f5f5f5;
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
}

.hero {
    text-align: center;
    padding: 3rem 1rem 2rem 1rem;
    animation: fadeUp 0.8s ease-in-out;
}

.title {
    font-size: 58px;
    font-weight: 900;
    letter-spacing: -2px;
    color: #ffffff;
}

.subtitle {
    color: #a3a3a3;
    font-size: 19px;
    margin-top: 12px;
}

.upload-area, .chat-area {
    background: #0f0f0f;
    border: 1px solid #262626;
    border-radius: 22px;
    padding: 26px;
    animation: fadeUp 1s ease-in-out;
}

.upload-area:hover, .chat-area:hover {
    border-color: #ffffff55;
    transition: 0.3s ease;
}

.metric {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 16px;
    padding: 18px;
    text-align: center;
}

.metric-value {
    font-size: 26px;
    font-weight: 900;
    color: #ffffff;
}

.metric-label {
    color: #9ca3af;
    font-size: 13px;
}

.answer {
    background: #ffffff;
    color: #050505;
    border-radius: 18px;
    padding: 24px;
    line-height: 1.7;
    font-size: 16px;
    animation: fadeUp 0.5s ease-in-out;
}

.chip {
    display: inline-block;
    padding: 9px 14px;
    margin: 6px 6px 6px 0;
    border-radius: 999px;
    background: #171717;
    border: 1px solid #2f2f2f;
    color: #e5e5e5;
    font-size: 14px;
}

.small {
    color: #9ca3af;
    font-size: 14px;
}

.stFileUploader {
    background: #111111;
    border: 1px dashed #3a3a3a;
    border-radius: 18px;
    padding: 12px;
}

.stTextInput input {
    background: #111111 !important;
    color: white !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 14px !important;
}

.stButton button {
    background: white;
    color: black;
    border-radius: 14px;
    border: none;
    font-weight: 800;
}

[data-testid="stSidebar"] {
    background: #090909;
    border-right: 1px solid #222;
}

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(18px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🧠 DocuMind AI")
    st.write("Ask questions from any text-based PDF.")
    st.divider()
    st.write("Works with:")
    st.write("• Notes")
    st.write("• Resumes")
    st.write("• Financial reports")
    st.write("• Research papers")
    st.write("• Articles")
    st.divider()
    st.caption("Scanned/image PDFs need OCR.")

st.markdown("""
<div class="hero">
    <div class="title">DocuMind AI</div>
    <div class="subtitle">Upload any text-based PDF. Ask questions. Get grounded answers.</div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1.5], gap="large")

with left:
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    st.subheader("Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        label_visibility="collapsed"
    )

    st.markdown("#### Suggested questions")
    for q in [
        "Summarize this document",
        "What are the key points?",
        "Explain this in simple words",
        "List the main topics",
        "What important details are mentioned?"
    ]:
        st.markdown(f'<span class="chip">{q}</span>', unsafe_allow_html=True)

    st.markdown('<p class="small">Best results with text-based PDFs.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="chat-area">', unsafe_allow_html=True)
    st.subheader("Ask your document")

    if uploaded_file is None:
        st.info("Upload a PDF to start.")

    else:
        with open("uploaded.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Reading document..."):
            text = extract_text("uploaded.pdf")

        if not text.strip():
            st.error("No readable text found. This may be a scanned/image PDF.")
            st.stop()

        with st.spinner("Creating document index..."):
            chunks = chunk_text(text)

            if len(chunks) == 0:
                st.error("No chunks created. Try another PDF.")
                st.stop()

            embeddings = create_embeddings(chunks)
            index = store_embeddings(embeddings)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric"><div class="metric-value">{len(text):,}</div><div class="metric-label">Characters</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric"><div class="metric-value">{len(chunks)}</div><div class="metric-label">Chunks</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown('<div class="metric"><div class="metric-value">RAG</div><div class="metric-label">Mode</div></div>', unsafe_allow_html=True)

        st.write("")

        question = st.text_input(
            "Ask a question",
            placeholder="Example: Summarize this PDF in simple words"
        )

        if question:
            with st.spinner("Thinking..."):
                retrieved_chunks = retrieve(question, index, chunks, top_k=10)
                context = "\\n".join(retrieved_chunks)

                if not context.strip():
                    st.error("No relevant context found. Try rephrasing.")
                    st.stop()

                answer = ask_llm(context, question)

            st.markdown("### Answer")
            st.markdown(f'<div class="answer">{answer}</div>', unsafe_allow_html=True)

            with st.expander("Retrieved Context"):
                st.write(context)

            with st.expander("Extracted Text Preview"):
                st.write(text[:2000])

    st.markdown('</div>', unsafe_allow_html=True)