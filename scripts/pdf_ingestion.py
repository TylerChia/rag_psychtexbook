from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# ---------- CONFIG ----------
PDF_PATH = '../data/Psyc Book N55.pdf'          # Path to your textbook
DB_DIR = "../data/vectorstore"             # Directory to save FAISS index
CHUNK_SIZE = 600                  # Characters per chunk
CHUNK_OVERLAP = 100                # Overlap to preserve context
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# ----------------------------


def ingest_pdf(pdf_path=PDF_PATH, db_dir=DB_DIR):
    print(f"📘 Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    print(f"🧩 Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks.")

    print(f"🧠 Creating embeddings with {EMBED_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print(f"💾 Building FAISS vectorstore...")
    db = FAISS.from_documents(chunks, embeddings)

    # Save to local directory
    os.makedirs(DB_DIR, exist_ok=True)
    db.save_local(DB_DIR)
    print(f"✅ Vectorstore saved at '{DB_DIR}'")

if __name__ == "__main__":
    ingest_pdf()