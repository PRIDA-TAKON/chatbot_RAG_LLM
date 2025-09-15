
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
import os

# --- Configuration ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"
LLM_MODEL_NAME = "google/flan-t5-small"

# --- Pydantic Models for API ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    source_documents: list = []

# --- Global Variables ---
app = FastAPI(title="RAG Chatbot API")
qa_chain = None

# --- Model Loading and Setup ---
@app.on_event("startup")
def startup_event():
    """This function runs when the API starts up. It loads all necessary models and sets up the QA chain."""
    global qa_chain
    print("--- API Startup: Loading models and setting up QA chain... ---")

    # 1. Load Embeddings and Vector Store
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}. Make sure the index directory is present.")
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever()

    # 2. Load the LLM
    print(f"Loading LLM: {LLM_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    llm = HuggingFacePipeline(pipeline=pipe)

    # 3. Create the RAG Chain
    template = '''Use the following context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context: {context}
    Question: {question}
    Answer:'''
    prompt_template = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )
    print("--- QA chain is ready. API startup complete. ---")

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "RAG Chatbot API is running. Use the /query endpoint to ask questions."}

@app.post("/query", response_model=QueryResponse)
def process_query(request: QueryRequest):
    """Receives a question, processes it through the RAG pipeline, and returns the answer."""
    if not qa_chain:
        raise HTTPException(status_code=503, detail="QA chain is not ready. The server might still be starting up.")

    print(f"Received query: {request.question}")
    response = qa_chain.invoke({"query": request.question})
    
    # Extract source documents and format them if needed
    source_docs_formatted = []
    if response.get("source_documents"):
        for doc in response["source_documents"]:
            source_docs_formatted.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })

    return {
        "answer": response.get("result", "Sorry, I could not find an answer."),
        "source_documents": source_docs_formatted
    }

# To run this API locally for testing:
# uvicorn backend_api:app --reload
