# PDF based QA system using RAG

This project is a **PDF Question-Answering system** built with:
- LangChain
- FAISS vector database
- HuggingFace embeddings
- Groq LLM API (free API key available)
- Streamlit UI

It lets users:
1. Upload a PDF
2. Convert PDF text into embeddings
3. Store/retrieve chunks using FAISS
4. Ask natural language questions and get answers grounded in the PDF content

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── chat.ipynb
```

## How It Works

1. Upload a PDF from the sidebar.
2. The app reads the PDF using `PyPDFLoader`.
3. Text is split into chunks with overlap for better retrieval.
4. Chunks are embedded via `sentence-transformers/all-MiniLM-L6-v2`.
5. Embeddings are indexed in a FAISS database.
6. At question time, relevant chunks are retrieved and passed to Groq LLM (`llama-3.1-8b-instant`) to generate an answer.

## Setup

### 1) Clone repository

```bash
git clone <your-repo-url>.git
cd "RAG Chatbot"
```

### 2) Create virtual environment

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Configure API key

- Get a free Groq API key from https://console.groq.com/keys
- Create `.env` from `.env.example`:

```bash
copy .env.example .env
```

- Open `.env` and add your key:

```env
GROQ_API_KEY=your_real_key_here
```

## Run the App

```bash
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## Usage

1. Upload any PDF file.
2. Click **Process PDF**.
3. Ask questions in natural language.
4. Review generated answer + retrieved source chunks.

## GitHub Upload Steps

Run these commands from project folder:

```bash
git init
git add .
git commit -m "Initial commit: PDF QA RAG system"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

## Notes

- `faiss_index/` and uploaded files are ignored in Git to keep repo clean.
- If you want multi-PDF support, extend the pipeline by merging chunks from multiple uploads before indexing.
