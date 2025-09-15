import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os # To manage file paths

def load_and_chunk_data(file_path="agnos_forum_data.json", chunk_size=1000, chunk_overlap=200):
    """Loads data from JSON, combines questions and answers, and chunks the text."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    for entry in data:
        question = entry.get('question', '')
        answers = entry.get('answers', [])
        
        # Combine question and answers into a single document string
        combined_text = f"คำถาม: {question}\n\nคำตอบ: {'\n\n'.join(answers)}\n\nURL: {entry.get('url', '')}"
        documents.append(combined_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = []
    for doc in documents:
        chunks.extend(text_splitter.split_text(doc))

    print(f"Loaded {len(data)} documents and created {len(chunks)} chunks.")
    return chunks

def create_and_save_vector_store(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2", faiss_index_path="faiss_index"):
    """Creates embeddings from chunks and builds a FAISS vector store."""
    print(f"Loading embedding model: {model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    print("Creating FAISS vector store...")
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    print(f"Saving FAISS index to {faiss_index_path}...")
    vector_store.save_local(faiss_index_path)
    print("FAISS index saved.")
    return vector_store

if __name__ == "__main__":
    chunks = load_and_chunk_data()
    # Optionally save chunks to a file for inspection
    with open("agnos_forum_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
    print("Chunks saved to agnos_forum_chunks.json")

    # Create and save the vector store
    vector_store = create_and_save_vector_store(chunks)
    print("Vector store creation complete.")
