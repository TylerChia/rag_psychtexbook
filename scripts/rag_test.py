# rag_query.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA

# 1Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the saved FAISS vector store
db = FAISS.load_local("../data/vectorstore/", embeddings, allow_dangerous_deserialization=True)

# Set up the local LLM 
llm = OllamaLLM(model="llama3.2:latest")  # or "llama3", "phi3", etc.

# Create a RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 5}),
    chain_type="stuff",
    return_source_documents=True
)

# Ask questions
while True:
    query = input("\nAsk a question (or 'exit'): ")
    if query.lower() in ["exit", "quit"]:
        break

    result = qa_chain.invoke({"query": query})
    
    # Answer
    print("\n🧩 Answer:\n", result["result"])
    
    # Source documents
    print("\n📄 Source Documents:")
    for i, doc in enumerate(result["source_documents"]):
        print(f"\n--- Document {i+1} ---")
        print(doc.page_content)  # the actual text
