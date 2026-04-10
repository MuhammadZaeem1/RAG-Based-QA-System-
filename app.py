import os
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq


# Load environment variables
load_dotenv()

INDEX_DIR = Path("faiss_index")
UPLOAD_DIR = Path("uploaded_docs")


def initialize_embeddings() -> HuggingFaceEmbeddings:
    """Create embedding model used to vectorize PDF chunks."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vector_store(pdf_path: Path) -> FAISS:
    """Load a PDF, split text into chunks, and create a FAISS vector index."""
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    embeddings = initialize_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(INDEX_DIR))
    return vector_store


def load_vector_store() -> Optional[FAISS]:
    """Load persisted FAISS index if available."""
    if not INDEX_DIR.exists():
        return None

    embeddings = initialize_embeddings()
    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_rerank_retriever(vector_store: FAISS) -> ContextualCompressionRetriever:
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 20})
    cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)
    return ContextualCompressionRetriever(base_retriever=base_retriever, base_compressor=compressor)


def get_llm() -> ChatGroq:
    """Initialize the LLM used for answer generation."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Add it to your environment or .env file.")

    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        groq_api_key=api_key,
        temperature=0.1,
    )


def answer_question(vector_store: FAISS, question: str) -> dict:
    """Run retrieval + generation and return answer with retrieved context."""
    llm = get_llm()

    retriever = build_rerank_retriever(vector_store)
    retrieved_docs = retriever.invoke(question)

    context_text = "\n\n".join(
        [
            f"[Source: {doc.metadata.get('source', 'Unknown source')} | Page: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
            for doc in retrieved_docs
        ]
    )

    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Use only the provided context to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "If the answer is not present in context, say you do not know."
    )
    chain = prompt | llm
    response = chain.invoke({"context": context_text, "question": question})

    return {
        "answer": response.content if hasattr(response, "content") else str(response),
        "context": retrieved_docs,
    }


def save_uploaded_pdf(uploaded_file) -> Path:
    """Persist uploaded Streamlit file to disk for processing."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_path = Path(tmp_file.name)

    final_path = UPLOAD_DIR / uploaded_file.name
    temp_path.replace(final_path)
    return final_path


def main() -> None:
    st.set_page_config(page_title="PDF QA with RAG", page_icon=":page_facing_up:", layout="wide")
    st.title("PDF based QA system using RAG")
    st.write("Upload a PDF, then ask natural language questions about its content.")

    with st.sidebar:
        st.header("1) Upload PDF")
        uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"])

        if st.button("Process PDF", type="primary"):
            if uploaded_pdf is None:
                st.warning("Please upload a PDF first.")
            else:
                with st.spinner("Reading and indexing the PDF..."):
                    pdf_path = save_uploaded_pdf(uploaded_pdf)
                    build_vector_store(pdf_path)
                st.success("PDF indexed successfully. You can now ask questions.")

    vector_store = load_vector_store()

    st.header("2) Ask Questions")
    question = st.text_input("Ask a question about your uploaded PDF")

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        if vector_store is None:
            st.warning("No PDF index found. Please upload and process a PDF first.")
            return

        try:
            with st.spinner("Generating answer..."):
                result = answer_question(vector_store, question)

            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Retrieved Context")
            for idx, doc in enumerate(result.get("context", []), start=1):
                source = doc.metadata.get("source", "Unknown source")
                page = doc.metadata.get("page", "N/A")
                st.markdown(f"**Chunk {idx}** | Source: {source} | Page: {page}")
                st.write(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
        except Exception as exc:
            st.error(f"Error: {exc}")


if __name__ == "__main__":
    main()
