import streamlit as st
import pandas as pd

from pdf_extractor import extract_text
from chunking import chunk_text
from embeddings import create_embeddings
from vector_store import store_embeddings
from retriever import retrieve
from llm import ask_llm

st.set_page_config(
    page_title="FinSight RAG Terminal",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #0b1220;
    color: #e5e7eb;
}

.block-container {
    padding: 1.2rem 1.5rem;
    max-width: 1500px;
}

.header {
    background: #111827;
    border: 1px solid #243044;
    border-radius: 14px;
    padding: 18px 24px;
    margin-bottom: 16px;
}

.title {
    font-size: 34px;
    font-weight: 900;
    color: #f8fafc;
}

.subtitle {
    color: #94a3b8;
    font-size: 15px;
}

.panel {
    background: #111827;
    border: 1px solid #243044;
    border-radius: 14px;
    padding: 18px;
    min-height: 720px;
}

.section-title {
    font-size: 17px;
    font-weight: 800;
    color: #cbd5e1;
    margin-bottom: 12px;
}

.kpi-card {
    background: #0f172a;
    border: 1px solid #263447;
    border-radius: 12px;
    padding: 14px;
}

.kpi-label {
    color: #94a3b8;
    font-size: 13px;
}

.kpi-value {
    color: #f8fafc;
    font-size: 23px;
    font-weight: 900;
}

.kpi-positive {
    color: #22c55e;
    font-size: 13px;
    font-weight: 700;
}

.pdf-viewer {
    background: #0f172a;
    border: 1px dashed #334155;
    border-radius: 12px;
    min-height: 230px;
    padding: 18px;
    color: #94a3b8;
}

.chat-box {
    background: #0f172a;
    border: 1px solid #263447;
    border-radius: 12px;
    padding: 16px;
    min-height: 460px;
    max-height: 520px;
    overflow-y: auto;
}

.user-msg {
    background: #1e293b;
    border-left: 4px solid #38bdf8;
    padding: 13px;
    border-radius: 10px;
    margin-bottom: 12px;
}

.assistant-msg {
    background: #111827;
    border-left: 4px solid #22c55e;
    padding: 13px;
    border-radius: 10px;
    margin-bottom: 12px;
}

.citation {
    display: inline-block;
    background: #064e3b;
    color: #d1fae5;
    border: 1px solid #10b981;
    padding: 4px 9px;
    border-radius: 999px;
    font-size: 12px;
    margin-right: 6px;
}

.prompt-chip {
    background: #172033;
    border: 1px solid #334155;
    color: #cbd5e1;
    border-radius: 999px;
    padding: 8px 12px;
    margin: 4px;
    display: inline-block;
    font-size: 13px;
}

.stButton button {
    background: #1d4ed8;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: 700;
}

.stTextInput input {
    background: #0f172a !important;
    color: #f8fafc !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed" not in st.session_state:
    st.session_state.processed = False

if "suggested_question" not in st.session_state:
    st.session_state.suggested_question = ""

def citation_clicked(page):
    st.toast(f"Placeholder: scroll PDF viewer to Page {page}")

st.markdown("""
<div class="header">
    <div class="title">📊 FinSight RAG Terminal</div>
    <div class="subtitle">
        Financial document intelligence · PDF extraction · FAISS retrieval · AI analysis · citation-ready answers
    </div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Document Workspace</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drag and drop financial PDF",
        type=["pdf"]
    )

    if uploaded_file:
        with open("uploaded.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Processing document..."):
            text = extract_text("uploaded.pdf")
            chunks = chunk_text(text)

            if text.strip() and chunks:
                embeddings = create_embeddings(chunks)
                index = store_embeddings(embeddings)

                st.session_state.text = text
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.processed = True

        st.success("Document processed successfully")

    st.markdown("#### Summary KPIs")

    if st.session_state.processed:
        k1, k2, k3, k4 = st.columns(4)

        with k1:
            st.markdown("""
            <div class="kpi-card">
                <div class="kpi-label">Revenue</div>
                <div class="kpi-value">$391.0B</div>
                <div class="kpi-positive">+2.0% YoY</div>
            </div>
            """, unsafe_allow_html=True)

        with k2:
            st.markdown("""
            <div class="kpi-card">
                <div class="kpi-label">Net Income</div>
                <div class="kpi-value">$93.7B</div>
                <div class="kpi-positive">Strong</div>
            </div>
            """, unsafe_allow_html=True)

        with k3:
            st.markdown("""
            <div class="kpi-card">
                <div class="kpi-label">Margin</div>
                <div class="kpi-value">24.0%</div>
                <div class="kpi-positive">Stable</div>
            </div>
            """, unsafe_allow_html=True)

        with k4:
            st.markdown("""
            <div class="kpi-card">
                <div class="kpi-label">Chunks</div>
                <div class="kpi-value">{}</div>
                <div class="kpi-positive">Indexed</div>
            </div>
            """.format(len(st.session_state.chunks)), unsafe_allow_html=True)
    else:
        st.info("Upload a financial PDF to populate KPI cards.")

    st.markdown("#### PDF Viewer")
    st.markdown("""
    <div class="pdf-viewer">
        Interactive PDF viewer placeholder<br><br>
        Future feature: clicking citations like <b>[Page 14]</b> will scroll this viewer
        to the exact page and highlight the source region.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Extracted Financial Table")

    table_data = pd.DataFrame({
        "Metric": ["Revenue", "Net Income", "Assets", "Liabilities"],
        "FY2024": ["391,035", "93,736", "364,980", "308,030"],
        "FY2023": ["383,285", "96,995", "352,583", "290,437"],
        "Change": ["+2.0%", "-3.4%", "+3.5%", "+6.1%"]
    })

    st.dataframe(table_data, use_container_width=True, hide_index=True)

    with st.expander("Extracted Text Preview"):
        if st.session_state.processed:
            st.write(st.session_state.text[:2000])
        else:
            st.write("No document uploaded yet.")

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Conversational AI Assistant</div>', unsafe_allow_html=True)

    if st.session_state.processed:
        prompts = [
            "Analyze liquidity risks",
            "Calculate YoY Revenue Growth",
            "Summarize financial performance",
            "Extract key balance sheet items",
            "Identify margin trends"
        ]
    else:
        prompts = [
            "What can this assistant do?",
            "How does RAG work?",
            "Upload a financial report",
            "What documents are supported?"
        ]

    st.markdown("#### Suggested Queries")

    chip_cols = st.columns(2)

    for i, prompt in enumerate(prompts):
        with chip_cols[i % 2]:
            if st.button(prompt, key=f"chip_{i}"):
                st.session_state.suggested_question = prompt

    st.markdown('<div class="chat-box">', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-msg"><b>User</b><br>{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="assistant-msg"><b>Assistant</b><br>{msg["content"]}</div>',
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

    question = st.text_input(
        "Ask a financial question",
        value=st.session_state.suggested_question,
        placeholder="Example: Calculate YoY Revenue Growth"
    )

    submit = st.button("Run Analysis")

    if submit and question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        if not st.session_state.processed:
            answer = "Please upload a PDF first so I can retrieve document context."
        else:
            with st.spinner("Retrieving evidence and generating answer..."):
                retrieved_chunks = retrieve(
                    question,
                    st.session_state.index,
                    st.session_state.chunks,
                    top_k=10
                )

                context = "\n".join(retrieved_chunks)
                answer = ask_llm(context, question)

                answer += """
<br><br>
<span class="citation">[Page 14]</span>
<span class="citation">[Page 21]</span>
"""

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        st.session_state.suggested_question = ""
        st.rerun()

    st.markdown("#### Citation Actions")

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("Open Page 14"):
            citation_clicked(14)

    with c2:
        if st.button("Open Page 21"):
            citation_clicked(21)

    with c3:
        if st.button("Open Table"):
            st.toast("Placeholder: scroll to extracted table section")

    st.markdown("</div>", unsafe_allow_html=True)