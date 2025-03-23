import requests
import os
from fastapi import FastAPI
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel

app = FastAPI()

# 🔹 Load FAISS index
faiss_index_path = "faiss_index"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.load_local(faiss_index_path, embeddings=embedding_model, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever()

# 🔹 Hugging Face API Details
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "YOUR_HUGGINGFACE_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_huggingface_api(prompt):
    """Sends a request to the Hugging Face API and handles errors."""
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 50,  # Limit to ensure concise answers
            "temperature": 0.5,  # Reduce randomness
            "top_p": 0.8  # Control diversity
        }
    }
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"].strip()
        return "Unexpected response format."
    except requests.exceptions.RequestException as e:
        return f"API Error: {str(e)}"

@app.get("/health")
def health_check():
    return {"status": "ok"}

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask_question(request: QueryRequest):
    query = request.query
    retrieved_docs = retriever.get_relevant_documents(query)

    if not retrieved_docs:
        response = "No relevant information found in the database."
    else:
        # Use only 1 document and limit its length
        context = " ".join([doc.page_content for doc in retrieved_docs[:1]])[:1000]  

        prompt = f"Context: {context}\nQuestion: {query}\nAnswer (only generate the final and precise solution,in one sentence):"

        response = query_huggingface_api(prompt)

    return {"answer": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
