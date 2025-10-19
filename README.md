# RAG Textbook Assistant
- Developed a Retrieval-Augmented Generation (RAG) system that enables users to upload textbooks or PDFs, automatically process them into searchable chunks, and generate semantic embeddings using FAISS and HuggingFace models.
- Leveraged LangChain to orchestrate document loading, text splitting, embedding creation, and retrieval-based question answering through a modular pipeline.
- Integrated a local LLM (via Ollama) within a Flask web interface, allowing users to ask natural language questions and receive context-grounded answers sourced directly from the uploaded documents.
- Built a modern, responsive UI with file upload functionality, dynamic vector store handling, and error messaging when no document context is available.


## Tech Stack
| Component                  | Description                                        |
| -------------------------- | -------------------------------------------------- |
| **LangChain**              | Framework for building RAG pipelines               |
| **FAISS**                  | Vector database for similarity search              |
| **HuggingFace Embeddings** | Generates vector representations of text           |
| **Ollama**                 | Local LLM runtime (e.g. `llama3`, `mistral`, etc.) |
| **Flask**                  | Lightweight backend + web UI                       |
| **HTML/CSS**               | Clean modern frontend for interactions             |


## How it Works
1. PDF Ingestion

- The uploaded textbook is read and split into overlapping text chunks.
- Each chunk is embedded using a HuggingFace model (all-MiniLM-L6-v2).
- The embeddings are stored in a FAISS vector database.

2. Question Answering

- When a question is asked, relevant chunks are retrieved via cosine similarity.
- These chunks are passed to the local LLM (via Ollama).
- LangChain’s RetrievalQA chain fuses the retrieved context with the user query.

3. Grounded Responses
- The LLM is instructed to answer only using textbook information.
- If the answer isn’t in the text, it replies with:
- “The answer is not available in the provided textbook.”