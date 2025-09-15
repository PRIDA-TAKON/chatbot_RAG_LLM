import os
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

# --- Configuration ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"
LLM_MODEL_NAME = "google/gemma-2b-it"

def setup_chatbot():
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
    if not os.path.exists(FAISS_INDEX_PATH):
        print(f"Error: FAISS index not found at {FAISS_INDEX_PATH}. Please run rag_pipeline.py first.")
        return None
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever()

    print(f"Loading LLM: {LLM_MODEL_NAME}...")
    # Ensure you have enough VRAM if not using CPU
    # For Gemma, you might need to specify device_map="auto" or "cpu"
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="cpu" # Force model to run on CPU
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.15
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # Define the prompt template for RAG
    template = """ใช้ข้อมูลบริบทต่อไปนี้เพื่อตอบคำถามที่ส่วนท้าย
    หากคุณไม่ทราบคำตอบ ให้บอกว่าคุณไม่ทราบ อย่าพยายามสร้างคำตอบขึ้นมาเอง

    {context}

    คำถาม: {question}
    คำตอบ:"""

    prompt = PromptTemplate.from_template(template)

    print("Setting up RAG chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # Stuffing all retrieved documents into the prompt
        retriever=retriever,
        return_source_documents=False, # Set to True if you want to see source chunks
        chain_type_kwargs={
            "prompt": prompt
        }
    )
    print("Chatbot setup complete.")
    return qa_chain

def chat_interface(qa_chain):
    print("\n--- Chatbot Ready ---\n")
    print("พิมพ์ 'exit' เพื่อออกจากการสนทนา")
    while True:
        user_input = input("คุณ: ")
        if user_input.lower() == 'exit':
            print("บอท: ลาก่อนครับ!")
            break
        
        print("บอท: กำลังคิด...")
        try:
            response = qa_chain.invoke({"query": user_input})
            print(f"บอท: {response['result']}")
        except Exception as e:
            print(f"บอท: เกิดข้อผิดพลาดในการประมวลผลคำถาม: {e}")

if __name__ == "__main__":
    qa_chain = setup_chatbot()
    if qa_chain:
        chat_interface(qa_chain)
