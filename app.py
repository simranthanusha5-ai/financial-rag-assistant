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
        radial-gradient(circle at 25% 10%, rgba(168,85,247,0.35), transparent 25%),
        radial-gradient(circle at 85% 20%, rgba(34,211,238,0.28), transparent 28%),
        linear-gradient(135deg, #020617 0%, #050816 45%, #0a1024 100%);
    color: #f8fafc;
}

.block-container {
    padding: 2rem 2.5rem;
    max-width: 1400px;
}

[data-testid="stSidebar"] {
    background: rgba(2, 6, 23, 0.92);
    border-right: 1px solid rgba(168,85,247,0.35);
}

.sidebar-title {
    font-size: 28px;
    font-weight: 900;
    background: linear-gradient(90deg, #a855f7, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.nav-item {
    padding: 14px 16px;
    margin: 10px 0;
    border-radius: 16px;
    color: #e2e8f0;
    background: rgba(15,23,42,0.65);
    border: 1px solid rgba(148,163,184,0.14);
}

.nav-active {
    background: linear-gradient(90deg, rgba(168,85,247,0.42), rgba(34,211,238,0.18));
    border: 1px solid rgba(168,85,247,0.65);
    box-shadow: 0 0 25px rgba(168,85,247,0.25);
}

.hero {
    padding: 34px;
    border-radius: 32px;
    background: rgba(2, 6, 23, 0.62);
    border: 1px solid rgba(34,211,238,0.25);
    box-shadow: 0 0 70px rgba(34,211,238,0.13), inset 0 0 30px rgba(255,255,255,0.03);
}

.hero-title {
    font-size: 58px;
    font-weight: 900;
    line-height: 1.05;
    background: linear-gradient(90deg, #ec4899, #a855f7, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 19px;
    color: #cbd5e1;
    margin-top: 14px;
}

.status-pill {
    display: inline-block;
    padding: 13px 22px;
    border-radius: 18px;
    background: rgba(15,23,42,0.78);
    border: 1px solid rgba(34,211,238,0.35);
    box-shadow: 0 0 30px rgba(34,211,238,0.13);
    text-align: center;
    font-weight: 700;
}

.card {
    padding: 26px;
    border-radius: 26px;
    background: rgba(2, 6, 23, 0.74);
    border: 1px solid rgba(148,163,184,0.18);
    box-shadow: 0 22px 80px rgba(0,0,0,0.4), 0 0 40px rgba(168,85,247,0.11);
}

.card-title {
    font-size: 18px;
    font-weight: 800;
    color: #c4b5fd;
    margin-bottom: 18px;
}

.upload-box {
    padding: 32px;
    text-align: center;
    border-radius: 22px;
    border: 1px dashed rgba(168,85,247,0.65);
    background: rgba(15,23,42,0.55);
    color: #cbd5e1;
}

.metric-card {
    padding: 20px;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(168,85,247,0.18), rgba(34,211,238,0.10));
    border: 1px solid rgba(34,211,238,0.25);
    box-shadow: inset 0 0 18px rgba(255,255,255,0.03);
}

.metric-number {
    font-size: 28px;
    font-weight: 900;
    color: #e9d5ff;
}

.answer-box {
    min-height: 190px;
    padding: 28px;
    border-radius: 24px;
    background:
        radial-gradient(circle at 85% 30%, rgba(168,85,247,0.25), transparent 25%),
        linear-gradient(135deg, rgba(15,23,42,0.92), rgba(2,6,23,0.88));
    border: 1px solid rgba(34,211,238,0.4);
    box-shadow: 0 0 45px rgba(34,211,238,0.15), inset 0 0 25px rgba(168,85,247,0.08);
    line-height: 1.7;
}

.tip-box {
    padding: 20px;
    border-radius: 20px;
    background: rgba(168,85,247,0.12);
    border: 1px solid rgba(168,85,247,0.35);
    color: #ddd6fe;
}

.stTextInput input {
    background: rgba(15,23,42,0.9) !important;
    color: white !important;
    border: 1px solid rgba(168,85,247,0.5) !important;
    border-radius: 18px !important;
    height: 58px;
}

.stFileUploader {
    background: rgba(15,23,42,0.55);
    border-radius: 18px;
    padding: 12px;
    border: 1px solid rgba(34,211,238,0.25);
}

div[data-testid="stExpander"] {
    background: rgba(2,6,23,0.72);
    border: 1px solid rgba(34,211,238,0.22);
    border-radius: 20px;
}

@keyframes glow {
    from { box-shadow: 0 0 25px rgba(168,85,247,0.18); }
    to { box-shadow: 0 0 55px rgba(34,211,238,0.22); }
}

.card, .hero {
    animation: glow 3s infinite alternate;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-title">🧠 RAG AI<br>Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item nav-active">💬 Chat</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">📤 Upload PDF</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">📄 Documents</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">📊 Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">⚙️ Settings</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="tip-box">
        <b>Powered by</b><br><br>
        🧊 RAG + LLM<br>
        🎯 FAISS Vector Search<br>
        ⚡ Groq LLM
    </div>
    """, unsafe_allow_html=True)

top_left, top_right = st.columns([1.4, 1])

with top_left:
    st.markdown("""
    <div class="hero">
        <div class="hero-title">AI Document<br>RAG Assistant</div>
        <div class="hero-subtitle">
            Upload a PDF and ask intelligent questions using AI + RAG.
        </div>
    </div>
    """, unsafe_allow_html=True)

with top_right:
    a, b = st.columns(2)
    with a:
        st.markdown('<div class="status-pill">🟢 System Online</div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="status-pill">⚡ Groq LLM</div>', unsafe_allow_html=True)

    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="status-pill">🔗<br>RAG Pipeline</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="status-pill">🎯<br>FAISS DB</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="status-pill">🛡️<br>Retrieval</div>', unsafe_allow_html=True)

st.write("")

left, right = st.columns([1, 1.9], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📤 Upload Document</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="upload-box">
        ☁️<br><br>
        Drag & drop your PDF here<br>
        or click to browse
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose PDF File", type=["pdf"], label_visibility="collapsed")

    st.markdown("#### ✨ Try asking")
    example_questions = [
        "What is this document about?",
        "Summarize the key points",
        "What technical skills are mentioned?",
        "Who received the certificate?",
        "What were the net sales?",
        "What are the main projects?"
    ]

    for q in example_questions:
        st.markdown(f'<div class="nav-item">{q} ➜</div>', unsafe_allow_html=True)

    st.markdown('<div class="tip-box">💡 Tip: The more specific your question, the better the answer.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">💬 Ask Your Question</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
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
            st.markdown(f'<div class="metric-card"><div>📄 Characters</div><div class="metric-number">{len(text):,}</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div>🧩 Chunks</div><div class="metric-number">{len(chunks)}</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown('<div class="metric-card"><div>🎯 Vector DB</div><div class="metric-number">FAISS</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown('<div class="metric-card"><div>⚡ Model</div><div class="metric-number">Groq</div></div>', unsafe_allow_html=True)

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

            st.markdown('<div class="card-title">✨ Answer</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

            with st.expander("🔍 Retrieved Context"):
                st.write(context)

            with st.expander("📃 Extracted Text Preview"):
                st.write(text[:1500])

    else:
        st.info("Upload a PDF to start asking questions.")

    st.markdown('</div>', unsafe_allow_html=True)