import streamlit as st
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

# Use Streamlit's caching to load the model and retriever only once.
@st.cache_resource
def setup_chatbot():
    """
    Loads the embedding model, FAISS index, and the LLM pipeline.
    This function is cached to avoid reloading everything on each interaction.
    """
    st.write("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    st.write(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
    if not os.path.exists(FAISS_INDEX_PATH):
        st.error(f"Error: FAISS index not found at {FAISS_INDEX_PATH}. Please run rag_pipeline.py first.")
        return None
    
    # allow_dangerous_deserialization is needed for FAISS with langchain.
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever()

    st.write(f"Loading LLM: {LLM_MODEL_NAME}...")
    
    # Configure quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto" # Automatically select device (CPU/GPU)
    )

    # Create a text generation pipeline
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

    # Define the prompt template
    template = '''ใช้ข้อมูลบริบทต่อไปนี้เพื่อตอบคำถามที่ส่วนท้าย
    หากคุณไม่ทราบคำตอบ ให้บอกว่าคุณไม่ทราบ อย่าพยายามสร้างคำตอบขึ้นมาเอง

    {context}

    คำถาม: {question}
    คำตอบ:'''
    prompt_template = PromptTemplate.from_template(template)

    st.write("Setting up RAG chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt_template}
    )
    
    st.success("Chatbot setup complete!")
    return qa_chain

# --- Streamlit UI ---

st.title("🤖 Chatbot RAG with Gemma-2B")
st.caption("ถาม-ตอบข้อมูลจากไฟล์เอกสารของคุณ")

# Load the QA chain
qa_chain = setup_chatbot()

if qa_chain:
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "สวัสดีครับ! ถามอะไรเกี่ยวกับเอกสารของคุณได้เลย"}]

    # Display past messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Get user input
    if prompt := st.chat_input("คำถามของคุณ..."):
        # Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("กำลังคิด..."):
                try:
                    response = qa_chain.invoke({"query": prompt})
                    result = response.get('result', "ขออภัย ผมไม่สามารถหาคำตอบได้ในขณะนี้")
                    st.write(result)
                    # Add assistant response to history
                    st.session_state.messages.append({"role": "assistant", "content": result})
                except Exception as e:
                    error_message = f"เกิดข้อผิดพลาด: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
else:
    st.warning("ไม่สามารถเริ่มต้น Chatbot ได้ กรุณาตรวจสอบว่าคุณได้รัน `rag_pipeline.py` เพื่อสร้าง index แล้ว")