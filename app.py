import streamlit as st

from pdf_extractor import extract_text
from chunking import chunk_text
from embeddings import create_embeddings
from vector_store import store_embeddings
from retriever import retrieve
from llm import ask_llm

st.set_page_config(
    page_title="AI Document RAG Assistant",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 20% 10%, rgba(168,85,247,0.28), transparent 26%),
        radial-gradient(circle at 90% 15%, rgba(34,211,238,0.22), transparent 28%),
        linear-gradient(135deg, #020617 0%, #050816 45%, #08111f 100%);
    color: #f8fafc;
}

.block-container {
    padding: 2rem 2.5rem;
    max-width: 1300px;
}

[data-testid="stSidebar"] {
    background: #050816;
    border-right: 1px solid rgba(168,85,247,0.28);
}

[data-testid="stSidebar"] * {
    color: #f8fafc;
}

.sidebar-logo {
    font-size: 30px;
    font-weight: 900;
    line-height: 1.1;
    margin-bottom: 28px;
    background: linear-gradient(90deg, #d946ef, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.side-item {
    padding: 14px 16px;
    margin: 10px 0;
    border-radius: 16px;
    background: rgba(15,23,42,0.75);
    border: 1px solid rgba(148,163,184,0.14);
    font-weight: 600;
}

.side-active {
    background: linear-gradient(90deg, rgba(168,85,247,0.45), rgba(34,211,238,0.14));
    border: 1px solid rgba(168,85,247,0.65);
    box-shadow: 0 0 26px rgba(168,85,247,0.22);
}

.power-card {
    margin-top: 28px;
    padding: 20px;
    border-radius: 22px;
    background: rgba(88,28,135,0.22);
    border: 1px solid rgba(168,85,247,0.35);
}

.hero-card {
    padding: 34px 38px;
    border-radius: 30px;
    background:
        radial-gradient(circle at 85% 20%, rgba(34,211,238,0.16), transparent 30%),
        rgba(2,6,23,0.78);
    border: 1px solid rgba(34,211,238,0.22);
    box-shadow: 0 24px 90px rgba(0,0,0,0.38);
}

.hero-title {
    font-size: 56px;
    font-weight: 900;
    letter-spacing: -1.8px;
    line-height: 1.03;
    background: linear-gradient(90deg, #ec4899, #a855f7, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-text {
    margin-top: 16px;
    font-size: 19px;
    color: #cbd5e1;
}

.status-card {
    min-height: 96px;
    padding: 20px;
    border-radius: 22px;
    background: rgba(2,6,23,0.76);
    border: 1px solid rgba(34,211,238,0.22);
    text-align: center;
    font-weight: 800;
    box-shadow: 0 18px 60px rgba(0,0,0,0.3);
}

.panel {
    padding: 28px;
    border-radius: 28px;
    background: rgba(2,6,23,0.78);
    border: 1px solid rgba(148,163,184,0.16);
    box-shadow: 0 22px 85px rgba(0,0,0,0.36);
}

.panel-title {
    font-size: 18px;
    font-weight: 900;
    color: #c4b5fd;
    margin-bottom: 18px;
}

.metric-box {
    padding: 20px;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(168,85,247,0.20), rgba(34,211,238,0.10));
    border: 1px solid rgba(34,211,238,0.22);
}

.metric-label {
    color: #cbd5e1;
    font-size: 14px;
    font-weight: 700;
}

.metric-value {
    margin-top: 8px;
    font-size: 28px;
    font-weight: 900;
    color: #f0abfc;
}

.answer-box {
    padding: 26px;
    border-radius: 24px;
    background:
        radial-gradient(circle at 90% 10%, rgba(168,85,247,0.22), transparent 30%),
        rgba(15,23,42,0.78);
    border: 1px solid rgba(34,211,238,0.32);
    border-left: 5px solid #22d3ee;
    line-height: 1.75;
    font-size: 16px;
}

.example {
    padding: 13px 15px;
    margin: 9px 0;
    border-radius: 14px;
    background: rgba(15,23,42,0.82);
    border: 1px solid rgba(148,163,184,0.14);
    color: #e9d5ff;
}

.tip {
    padding: 18px;
    border-radius: 18px;
    background: rgba(168,85,247,0.14);
    border: 1px solid rgba(168,85,247,0.3);
    color: #ddd6fe;
}

.stFileUploader {
    padding: 14px;
    border-radius: 18px;
    border: 1px dashed rgba(168,85,247,0.5);
    background: rgba(15,23,42,0.58);
}

.stTextInput input {
    height: 58px;
    border-radius: 16px !important;
    background: rgba(15,23,42,0.92) !important;
    color: white !important;
    border: 1px solid rgba(34,211,238,0.35) !important;
}

div[data-testid="stExpander"] {
    background: rgba(2,6,23,0.72);
    border: 1px solid rgba(34,211,238,0.18);
    border-radius: 18px;
}

hr {
    border-color: rgba(148,163,184,0.18);
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-logo">🧠 RAG AI<br>Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-item side-active">💬 Chat</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-item">📤 Upload PDF</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-item">📄 Documents</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-item">📊 Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-item">⚙️ Settings</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="power-card">
        <b>Powered by</b><br><br>
        🧊 RAG + LLM<br>
        🎯 FAISS Vector Search<br>
        ⚡ Groq LLM
    </div>
    """, unsafe_allow_html=True)

top1, top2 = st.columns([1.55, 1], gap="large")

with top1:
    st.markdown("""
    <div class="hero-card">
        <div class="hero-title">AI Document<br>RAG Assistant</div>
        <div class="hero-text">
            Upload a PDF and ask intelligent questions using semantic retrieval and LLM reasoning.
        </div>
    </div>
    """, unsafe_allow_html=True)

with top2:
    s1, s2 = st.columns(2)
    with s1:
        st.markdown('<div class="status-card">🟢<br>System Online</div>', unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="status-card">⚡<br>Groq LLM</div>', unsafe_allow_html=True)

    st.write("")
    s3, s4, s5 = st.columns(3)
    with s3:
        st.markdown('<div class="status-card">🔗<br>RAG</div>', unsafe_allow_html=True)
    with s4:
        st.markdown('<div class="status-card">🎯<br>FAISS</div>', unsafe_allow_html=True)
    with s5:
        st.markdown('<div class="status-card">🛡️<br>Grounded</div>', unsafe_allow_html=True)

st.write("")

left, right = st.columns([1, 1.75], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📤 Upload Document</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose PDF File",
        type=["pdf"],
        label_visibility="collapsed"
    )

    st.caption("Max file size: 200MB · PDF only")

    st.markdown("#### ✨ Try asking")
    for q in [
        "What is this document about?",
        "Summarize the key points",
        "What technical skills are mentioned?",
        "Who received the certificate?",
        "What were the net sales?",
        "What are the main projects?"
    ]:
        st.markdown(f'<div class="example">{q} →</div>', unsafe_allow_html=True)

    st.markdown('<div class="tip">💡 Tip: Ask specific questions for better answers.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">💬 Ask Your Question</div>', unsafe_allow_html=True)

    if uploaded_file is None:
        st.info("Upload a PDF to start asking questions.")
    else:
        with open("uploaded.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("✅ Document uploaded successfully")

        with st.spinner("Reading document..."):
            text = extract_text("uploaded.pdf")

        if not text.strip():
            st.error("No readable text found. This PDF may be scanned/image-based. OCR is needed.")
            st.stop()

        with st.spinner("Creating chunks, embeddings, and FAISS index..."):
            chunks = chunk_text(text)

            if len(chunks) == 0:
                st.error("No text chunks created. Please upload a text-based PDF.")
                st.stop()

            embeddings = create_embeddings(chunks)
            index = store_embeddings(embeddings)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-box"><div class="metric-label">📄 Characters</div><div class="metric-value">{len(text):,}</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-box"><div class="metric-label">🧩 Chunks</div><div class="metric-value">{len(chunks)}</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown('<div class="metric-box"><div class="metric-label">🎯 Vector DB</div><div class="metric-value">FAISS</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown('<div class="metric-box"><div class="metric-label">⚡ Model</div><div class="metric-value">Groq</div></div>', unsafe_allow_html=True)

        st.write("")
        question = st.text_input(
            "Ask a question",
            placeholder="Type your question here...",
            label_visibility="collapsed"
        )

        if question:
            with st.spinner("Generating grounded answer..."):
                retrieved_chunks = retrieve(question, index, chunks, top_k=8)
                context = "\n".join(retrieved_chunks)

                if not context.strip():
                    st.error("No relevant context retrieved. Try rephrasing the question.")
                    st.stop()

                answer = ask_llm(context, question)

                if answer.strip().lower().replace(".", "") == "not found":
                    q = question.lower()
                    if "who" in q and "certificate" in q and "issued on" in context.lower():
                        answer = context.split("Issued on:")[0].strip()

            st.markdown("### ✨ Answer")
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

            with st.expander("🔍 Retrieved Context"):
                st.write(context)

            with st.expander("📃 Extracted Text Preview"):
                st.write(text[:1500])

    st.markdown('</div>', unsafe_allow_html=True)