import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA

VECTOR_DIR = "data/vectorstore"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def ingest_pdf(pdf_path: str):
    """Load a PDF, chunk it, embed it, and save to FAISS."""
    print(f"📘 Ingesting PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    print(f"🧠 Creating embeddings with {EMBED_MODEL}...")
    db = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTOR_DIR, exist_ok=True)
    db.save_local(VECTOR_DIR)
    print("✅ Vectorstore updated.")


def get_qa_chain():
    """Load vectorstore + LLM into a RetrievalQA pipeline."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Check if the vectorstore directory and required files exist
    if not os.path.exists(VECTOR_DIR):
        raise FileNotFoundError(
            f"❌ No vector store found at '{VECTOR_DIR}'. "
            f"Please upload and ingest a PDF first."
        )

    faiss_index = os.path.join(VECTOR_DIR, "index.faiss")
    faiss_meta = os.path.join(VECTOR_DIR, "index.pkl")

    if not (os.path.exists(faiss_index) and os.path.exists(faiss_meta)):
        raise FileNotFoundError(
            f"⚠️ Vector store incomplete — missing 'index.faiss' or 'index.pkl' in {VECTOR_DIR}. "
            f"Please re-run the PDF ingestion process."
        )

    # Load FAISS and build the QA pipeline
    db = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
    llm = OllamaLLM(model="llama3.2:latest")

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 5}),
        chain_type="stuff",
        return_source_documents=True
    )

