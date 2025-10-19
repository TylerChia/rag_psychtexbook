from flask import Flask, render_template, request, jsonify
import os
from rag_utils import ingest_pdf, get_qa_chain

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

qa_chain = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_pdf():
    global qa_chain
    if "pdf" not in request.files:
        return jsonify({"error": "No PDF uploaded."}), 400

    pdf_file = request.files["pdf"]
    pdf_path = os.path.join(UPLOAD_FOLDER, pdf_file.filename)
    pdf_file.save(pdf_path)

    # Ingest the new PDF and rebuild vectorstore
    ingest_pdf(pdf_path)
    qa_chain = get_qa_chain()

    return jsonify({"message": "✅ PDF uploaded and indexed successfully!"})


@app.route("/ask", methods=["POST"])
def ask():
    global qa_chain
    query = request.json.get("query", "")
    if not query:
        return jsonify({"error": "Empty query"}), 400

    # Try loading the QA chain safely
    try:
        if qa_chain is None:
            qa_chain = get_qa_chain()
    except:
        return jsonify({"error": "No Uploaded Textbook!"}), 400

    # Run query if we have a valid chain
    result = qa_chain.invoke({"query": query})
    answer = result["result"]
    sources = [doc.page_content[:300] + "..." for doc in result["source_documents"]]

    return jsonify({"answer": answer, "sources": sources})


if __name__ == "__main__":
    app.run(debug=True)
