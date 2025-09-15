import streamlit as st
import os
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

# --- Configuration ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"
LLM_MODEL_NAME = "google/flan-t5-small"

@st.cache_resource
def setup_chatbot():
    """
    Loads the embedding model, FAISS index, and the LLM pipeline.
    """
    st.write("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    st.write(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
    if not os.path.exists(FAISS_INDEX_PATH):
        st.error(f"Error: FAISS index not found at {FAISS_INDEX_PATH}. Please ensure it's in the GitHub repository.")
        return None
    
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever()

    st.write(f"Loading LLM: {LLM_MODEL_NAME}...")
    
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    template = '''Use the following context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}
    Answer:'''
    prompt_template = PromptTemplate.from_template(template)

    st.write("Setting up RAG chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,  # Set to True to get source documents
        chain_type_kwargs={"prompt": prompt_template}
    )
    
    st.success("Chatbot setup complete!")
    return qa_chain

# --- Streamlit UI ---

st.title("🤖 Chatbot RAG")
st.caption(f"Powered by {LLM_MODEL_NAME}")

qa_chain = setup_chatbot()

if qa_chain:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "สวัสดีครับ! ถามอะไรเกี่ยวกับเอกสารของคุณได้เลย"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("คำถามของคุณ..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นหาข้อมูลและสร้างคำตอบ..."):
                try:
                    response = qa_chain.invoke({"query": prompt})
                    result = response.get('result', "ขออภัย ผมไม่สามารถหาคำตอบได้ในขณะนี้")
                    
                    with st.expander("📚 เอกสารอ้างอิงที่ค้นเจอ (สำหรับดีบัก)"):
                        source_docs = response.get('source_documents', [])
                        if source_docs:
                            for doc in source_docs:
                                st.write(f"**Source:** `{doc.metadata.get('source', 'N/A')}`")
                                st.write(f"```\n{doc.page_content}\n```")
                                st.divider()
                        else:
                            st.write("ไม่พบเอกสารอ้างอิง")

                    st.write(result)
                    
                    st.session_state.messages.append({"role": "assistant", "content": result})

                except Exception as e:
                    error_message = f"เกิดข้อผิดพลาด: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
else:
    st.warning("ไม่สามารถเริ่มต้น Chatbot ได้ กรุณาตรวจสอบ Log ด้านบน")