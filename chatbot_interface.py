import streamlit as st
import requests
import os

# --- Configuration ---
# The URL of the backend API. 
# When running locally, this will be http://localhost:8000
# When deployed on AWS, you will need to replace this with your actual API Gateway URL.
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "https://PRIDA-RAGLLM.hf.space")

# --- Streamlit UI ---
st.title("🤖 Chatbot RAG (Frontend)")
st.caption(f"Connecting to backend at: {BACKEND_API_URL}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "สวัสดีครับ! ผมพร้อมรับคำถามแล้ว"}]

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get user input
if prompt := st.chat_input("คำถามของคุณ..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Call the backend API and display the response
    with st.chat_message("assistant"):
        with st.spinner("กำลังส่งคำถามไปที่ Backend..."):
            try:
                # Define the payload for the POST request
                payload = {"question": prompt}
                
                # Send the request to the backend API
                response = requests.post(BACKEND_API_URL, json=payload)
                response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx) 
                
                api_response = response.json()
                answer = api_response.get("answer", "ขออภัย, ไม่ได้รับคำตอบจาก API")

                # Display the answer
                st.write(answer)

                # Optionally, display source documents in an expander
                with st.expander("📚 เอกสารอ้างอิง"):
                    source_docs = api_response.get('source_documents', [])
                    if source_docs:
                        for doc in source_docs:
                            st.write(f"**Source:** `{doc['metadata'].get('source', 'N/A')}`")
                            st.write(f"```\n{doc['page_content']}\n```")
                            st.divider()
                    else:
                        st.write("ไม่พบเอกสารอ้างอิง")

                # Add assistant's response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except requests.exceptions.RequestException as e:
                error_message = f"ไม่สามารถเชื่อมต่อกับ Backend API ได้: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
            except Exception as e:
                error_message = f"เกิดข้อผิดพลาดที่ไม่คาดคิด: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})