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
    background:
        radial-gradient(circle at top left, rgba(255,255,255,0.07), transparent 28%),
        radial-gradient(circle at bottom right, rgba(255,255,255,0.04), transparent 30%),
        #050505;
    color: #f5f5f5;
}

.block-container {
    max-width: 1180px;
    padding-top: 1.2rem;
}

[data-testid="stSidebar"] {
    background: #090909;
    border-right: 1px solid #1f1f1f;
}

.hero {
    text-align: center;
    padding: 2rem 1rem 1.4rem 1rem;
    animation: fadeUp 0.8s ease-in-out;
}

.hero-title {
    font-size: 62px;
    font-weight: 900;
    letter-spacing: -2.5px;
    color: #ffffff;
}

.hero-subtitle {
    color: #a3a3a3;
    font-size: 19px;
    margin-top: 10px;
}

.hero-badges {
    margin-top: 18px;
}

.badge {
    display: inline-block;
    padding: 8px 14px;
    margin: 5px;
    border-radius: 999px;
    background: #111111;
    border: 1px solid #2b2b2b;
    color: #d4d4d4;
    font-size: 13px;
    transition: all 0.25s ease;
}

.badge:hover {
    border-color: #ffffff;
    box-shadow: 0 0 18px rgba(255,255,255,0.14);
    transform: translateY(-2px);
}

.upload-section, .ask-section {
    background: rgba(15, 15, 15, 0.94);
    border: 1px solid #262626;
    border-radius: 24px;
    padding: 26px;
    box-shadow: 0 20px 55px rgba(0,0,0,0.35);
    animation: fadeUp 1s ease-in-out;
    transition: all 0.25s ease;
}

.upload-section:hover, .ask-section:hover {
    border-color: #ffffff55;
    box-shadow: 0 0 30px rgba(255,255,255,0.10);
    transform: translateY(-2px);
}

.section-title {
    font-size: 24px;
    font-weight: 850;
    margin-bottom: 16px;
}

.metric {
    background: #111111;
    border: 1px solid #2b2b2b;
    border-radius: 18px;
    padding: 18px;
    text-align: center;
    transition: all 0.25s ease;
}

.metric:hover {
    border-color: #ffffff;
    box-shadow: 0 0 22px rgba(255,255,255,0.12);
    transform: translateY(-2px);
}

.metric-value {
    font-size: 26px;
    font-weight: 900;
    color: #ffffff;
}

.metric-label {
    color: #a3a3a3;
    font-size: 13px;
    margin-top: 4px;
}

.answer {
    background: #ffffff;
    color: #050505;
    border-radius: 20px;
    padding: 24px;
    line-height: 1.75;
    font-size: 16px;
    box-shadow: 0 12px 35px rgba(255,255,255,0.08);
    animation: fadeUp 0.5s ease-in-out;
}

.empty-answer {
    background: #111111;
    border: 1px dashed #3a3a3a;
    color: #a3a3a3;
    border-radius: 20px;
    padding: 35px;
    text-align: center;
}

.example-box {
    background: #111111;
    border: 1px solid #2b2b2b;
    border-radius: 14px;
    padding: 12px 14px;
    margin-bottom: 10px;
    color: #e5e5e5;
    font-size: 14px;
    transition: all 0.25s ease;
}

.example-box:hover {
    border-color: #ffffff;
    box-shadow: 0 0 20px rgba(255,255,255,0.11);
    transform: translateX(4px);
}

.small {
    color: #9ca3af;
    font-size: 14px;
}

.stFileUploader {
    background: #111111;
    border: 1px dashed #404040;
    border-radius: 18px;
    padding: 12px;
    margin-top: 0 !important;
    transition: all 0.25s ease;
}

.stFileUploader:hover {
    border-color: #ffffff;
    box-shadow: 0 0 20px rgba(255,255,255,0.10);
}

.stTextInput {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

.stTextInput input {
    background: #111111 !important;
    color: #ffffff !important;
    border: 1px solid #404040 !important;
    border-radius: 14px !important;
    height: 50px;
}

.stTextInput input:focus {
    border-color: #ffffff !important;
    box-shadow: 0 0 18px rgba(255,255,255,0.12) !important;
}

.stButton > button {
    width: 100%;
    background: #ffffff;
    color: #050505;
    border-radius: 14px;
    border: none;
    font-weight: 800;
    padding: 0.65rem 1rem;
}

.stButton > button:hover {
    background: #d4d4d4;
    color: #000000;
    transform: translateY(-1px);
    transition: 0.2s ease;
}

div[data-testid="stExpander"] {
    background: #0f0f0f;
    border: 1px solid #252525;
    border-radius: 14px;
}

/* remove unwanted labels / empty bars */
div[data-testid="stTextInput"] > label {
    display: none !important;
}

div[data-testid="stFileUploader"] > label {
    display: none !important;
}

section[data-testid="stFileUploaderDropzone"] {
    background: #111111 !important;
    border: 1px dashed #404040 !important;
    border-radius: 18px !important;
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

if "question_text" not in st.session_state:
    st.session_state.question_text = ""

if "answer" not in st.session_state:
    st.session_state.answer = ""

if "context" not in st.session_state:
    st.session_state.context = ""

with st.sidebar:
    st.title("🧠 DocuMind AI")
    st.write("Ask questions from any text-based PDF.")
    st.divider()
    st.write("**Supported documents**")
    st.write("• Study notes")
    st.write("• Resumes")
    st.write("• Financial reports")
    st.write("• Research papers")
    st.write("• Articles")
    st.divider()
    st.caption("Scanned/image PDFs need OCR.")

st.markdown("""
<div class="hero">
    <div class="hero-title">DocuMind AI</div>
    <div class="hero-subtitle">Upload any text-based PDF. Ask questions. Get grounded answers.</div>
    <div class="hero-badges">
        <span class="badge">RAG</span>
        <span class="badge">FAISS Vector Search</span>
        <span class="badge">Groq LLM</span>
        <span class="badge">PDF Intelligence</span>
    </div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1.5], gap="large")

with left:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Upload PDF</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )

    st.markdown('<p class="small">Best results with text-based PDFs.</p>', unsafe_allow_html=True)

    st.markdown("#### Suggested prompts")

    if uploaded_file is None:
        prompts = [
            "What can this assistant do?",
            "What type of PDFs are supported?",
            "How does RAG work?"
        ]
    else:
        prompts = [
            "Summarize this document in simple English",
            "What are the key points?",
            "List the main topics covered",
            "Explain the important concepts",
            "Create 5 exam questions from this PDF",
            "Give me a short revision summary"
        ]

    for i, prompt in enumerate(prompts):
        if st.button(prompt, key=f"prompt_{i}"):
            st.session_state.question_text = prompt
            st.session_state.answer = ""
            st.session_state.context = ""

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="ask-section">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Ask your document</div>', unsafe_allow_html=True)

    if uploaded_file is None:
        st.info("Upload a PDF to start.")
        st.markdown(
            '<div class="empty-answer">Your answer will appear here after you upload a PDF and ask a question.</div>',
            unsafe_allow_html=True
        )

    else:
        with open("uploaded.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Reading document..."):
            text = extract_text("uploaded.pdf")

        if not text.strip():
            st.error("No readable text found. This may be a scanned/image PDF.")
            st.stop()

        with st.spinner("Creating RAG index..."):
            chunks = chunk_text(text)

            if len(chunks) == 0:
                st.error("No chunks created. Try another PDF.")
                st.stop()

            embeddings = create_embeddings(chunks)
            index = store_embeddings(embeddings)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f'<div class="metric"><div class="metric-value">{len(text):,}</div><div class="metric-label">Characters</div></div>',
                unsafe_allow_html=True
            )
        with m2:
            st.markdown(
                f'<div class="metric"><div class="metric-value">{len(chunks)}</div><div class="metric-label">Chunks</div></div>',
                unsafe_allow_html=True
            )
        with m3:
            st.markdown(
                '<div class="metric"><div class="metric-value">Ready</div><div class="metric-label">RAG Status</div></div>',
                unsafe_allow_html=True
            )

        question = st.text_input(
            "Ask a question",
            value=st.session_state.question_text,
            placeholder="Example: Summarize this PDF in simple words",
            label_visibility="collapsed"
        )

        col_a, col_b = st.columns([1, 4])
        with col_a:
            run = st.button("Ask")
        with col_b:
            st.caption("Tip: Ask specific questions for better answers.")

        if run and question:
            with st.spinner("Thinking..."):
                retrieved_chunks = retrieve(question, index, chunks, top_k=10)
                context = "\n".join(retrieved_chunks)

                if not context.strip():
                    st.error("No relevant context found. Try rephrasing.")
                    st.stop()

                answer = ask_llm(context, question)

                st.session_state.answer = answer
                st.session_state.context = context

        st.markdown("### Answer")

        if st.session_state.answer:
            st.markdown(
                f'<div class="answer">{st.session_state.answer}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="empty-answer">Ask a question to generate an answer.</div>',
                unsafe_allow_html=True
            )

        with st.expander("Retrieved Context"):
            st.write(st.session_state.context if st.session_state.context else "No context retrieved yet.")

        with st.expander("Extracted Text Preview"):
            st.write(text[:2500])

    st.markdown("</div>", unsafe_allow_html=True)