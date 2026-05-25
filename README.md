# 🤖 Financial RAG Assistant

A Retrieval-Augmented Generation (RAG) based AI assistant that allows users to upload PDFs and ask intelligent questions from the document using local LLM inference.

---

## 🚀 Features

- Upload PDF documents
- Extract text from PDFs
- Split text into chunks
- Generate semantic embeddings
- Store embeddings in FAISS Vector Database
- Retrieve relevant document context
- Use local Qwen2.5 model through Ollama
- Answer document-based questions
- Show retrieved context
- Modern Streamlit UI with animations

Supports:
- Financial Reports
- Resumes
- Research Papers
- General Text PDFs

---

## 🏗 Architecture

```text
PDF Upload
↓
Text Extraction (PyMuPDF)
↓
Chunking
↓
Embeddings (Sentence Transformers)
↓
FAISS Vector DB
↓
Retriever
↓
Qwen Local LLM (Ollama)
↓
Structured Answer
```

---

## 🛠 Tech Stack

- Python
- Streamlit
- PyMuPDF
- LangChain Text Splitters
- Sentence Transformers
- FAISS
- Ollama
- Qwen2.5 3B

---

## 📂 Project Structure

```bash
financial-rag/
│
├── app.py
├── pdf_extractor.py
├── chunking.py
├── embeddings.py
├── vector_store.py
├── retriever.py
├── llm.py
├── test.py
├── .gitignore
└── README.md
```

---

## ▶️ Run Locally

### 1. Clone Repository

```bash
git clone https://github.com/simranthanusha5-ai/financial-rag-assistant.git
cd financial-rag-assistant
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Local LLM

```bash
ollama pull qwen2.5:3b
```

### 5. Run Streamlit App

```bash
streamlit run app.py
```

---

## 💡 Example Questions

### Financial PDFs
- What were Apple’s net sales?
- What are total assets?
- What are liabilities?

### Resume PDFs
- What technical skills are mentioned?
- Summarize work experience
- What AI/ML projects are listed?
- What certifications does the candidate have?

### General PDFs
- Summarize this document
- What are the key findings?
- What technologies are mentioned?

---

## 🔥 Why RAG?

RAG (Retrieval-Augmented Generation) reduces hallucinations by retrieving the most relevant document chunks before sending them to the LLM.

Benefits:
- Better factual grounding
- Higher accuracy
- Works on private/local documents
- No retraining needed
- Faster document QA

---

## ⚠ Current Limitations

- Best for text-based PDFs
- OCR for scanned/image PDFs not added yet
- Local inference depends on Ollama

---

## 👩‍💻 Author

**Thanusha Simran**  
AI & ML Undergraduate  
Aspiring Software Engineer