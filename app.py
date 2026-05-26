import streamlit as st

from pdf_extractor import extract_text
from chunking import chunk_text
from embeddings import create_embeddings
from vector_store import store_embeddings
from retriever import retrieve
from llm import ask_llm

st.set_page_config(page_title="DocuMind AI", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.stApp {
    background: #050505;
    color: white;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
}

[data-testid="stSidebar"] {
    background: #090909;
    border-right: 1px solid #222;
}

.title {
    text-align: center;
    font-size: 58px;
    font-weight: 900;
    color: white;
}

.subtitle {
    text-align: center;
    color: #aaa;
    font-size: 19px;
    margin-bottom: 22px;
}

.badge {
    display: inline-block;
    padding: 8px 14px;
    margin: 5px;
    border-radius: 999px;
    border: 1px solid #333;
    background: #111;
    color: #ddd;
    transition: 0.25s;
}

.badge:hover {
    border-color: white;
    box-shadow: 0 0 16px rgba(255,255,255,0.15);
    transform: translateY(-2px);
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #101010;
    border: 1px solid #2a2a2a;
    border-radius: 22px;
    padding: 12px;
    transition: 0.25s;
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #666;
    box-shadow: 0 0 25px rgba(255,255,255,0.09);
    transform: translateY(-2px);
}

.metric-box {
    background: #151515;
    border: 1px solid #2d2d2d;
    border-radius: 16px;
    padding: 18px;
    text-align: center;
}

.metric-value {
    font-size: 25px;
    font-weight: 900;
}

.answer {
    background: white;
    color: black;
    border-radius: 18px;
    padding: 22px;
    line-height: 1.7;
}

.empty-answer {
    background: #111;
    border: 1px dashed #333;
    color: #aaa;
    border-radius: 18px;
    padding: 30px;
    text-align: center;
}

.stButton > button {
    width: 100%;
    background: white;
    color: black;
    border-radius: 14px;
    border: none;
    font-weight: 800;
}

.stButton > button:hover {
    background: #ddd;
    transform: translateY(-1px);
}

.stTextInput input {
    background: #111 !important;
    color: white !important;
    border: 1px solid #444 !important;
    border-radius: 14px !important;
}

.stFileUploader {
    background: #111;
    border: 1px dashed #444;
    border-radius: 16px;
    padding: 10px;
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
    st.write("**Supported**")
    st.write("• Notes")
    st.write("• Resumes")
    st.write("• Reports")
    st.write("• Research papers")
    st.write("• Articles")
    st.divider()
    st.caption("Scanned PDFs need OCR.")

st.markdown('<div class="title">DocuMind AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload any text-based PDF. Ask questions. Get grounded answers.</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div style="text-align:center; margin-bottom:30px;">'
    '<span class="badge">RAG</span>'
    '<span class="badge">FAISS Vector Search</span>'
    '<span class="badge">Groq LLM</span>'
    '<span class="badge">PDF Intelligence</span>'
    '</div>',
    unsafe_allow_html=True
)

left, right = st.columns([1, 1.5], gap="large")

with left:
    with st.container(border=True):
        st.subheader("Upload PDF")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"]
        )

        st.caption("Best results with text-based PDFs.")

        st.markdown("#### Suggested prompts")

        prompts = [
            "Summarize this document in simple English",
            "What are the key points?",
            "List the main topics covered",
            "Explain the important concepts",
            "Create 5 exam questions from this PDF",
            "Give me a short revision summary",
        ]

        for i, prompt in enumerate(prompts):
            if st.button(prompt, key=f"prompt_{i}"):
                st.session_state.question_text = prompt
                st.session_state.answer = ""
                st.session_state.context = ""

with right:
    with st.container(border=True):
        st.subheader("Ask your document")

        if uploaded_file is None:
            st.info("Upload a PDF to start.")
            st.markdown(
                '<div class="empty-answer">Your answer will appear here.</div>',
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
                    '<div class="metric-box"><div class="metric-value">Ready</div>RAG Status</div>',
                    unsafe_allow_html=True
                )

            question = st.text_input(
                "Ask a question",
                value=st.session_state.question_text,
                placeholder="Example: Summarize this PDF"
            )

            run = st.button("Ask")

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
                st.write(st.session_state.context or "No context retrieved yet.")

            with st.expander("Extracted Text Preview"):
                st.write(text[:2500])