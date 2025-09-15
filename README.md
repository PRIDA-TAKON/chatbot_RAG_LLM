# Chatbot RAG with LLM

This project implements a Retrieval-Augmented Generation (RAG) chatbot. It uses a language model to answer questions based on a provided knowledge base, which is built from the `agnos_forum_data.json` file.

The application is designed with a decoupled architecture, separating the user interface (Frontend) from the AI processing (Backend) for scalability and performance.

---

## 🏛️ Architecture

The system consists of two main components:

1.  **Backend (FastAPI):** 
    - A Python application built with FastAPI.
    - Responsible for all heavy lifting: loading the embedding and LLM models, creating the FAISS vector store, and running the RAG inference pipeline.
    - Exposes a single API endpoint (`/query`) to receive questions and return AI-generated answers.
    - Designed to be containerized with Docker and deployed on a cloud service like Hugging Face Spaces.

2.  **Frontend (Streamlit):**
    - A lightweight Python application built with Streamlit.
    - Provides a simple, clean chat interface for the user.
    - Does **not** load any models or perform any AI processing.
    - When a user asks a question, it makes an HTTP request to the Backend API, receives the answer, and displays it.
    - Designed to be deployed on Streamlit Community Cloud.

### Flow Diagram

```
[User] <--> [Streamlit Frontend (UI)] <--> [FastAPI Backend (API on Hugging Face Spaces)] <--> [LLM & Vector Store]
```

---

## 🚀 Getting Started

### 1. Initial Setup

First, you need to prepare the knowledge base and install dependencies.

-   **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd 04-chatbot_RAG_LLM
    ```

-   **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

-   **Create the Vector Store:**
    Make sure the `agnos_forum_data.json` file is in the root directory. Then, run the `rag_pipeline.py` script **once** to create the `faiss_index` directory.
    ```bash
    python rag_pipeline.py
    ```

### 2. Running Locally for Testing

To test the full application on your local machine, you need to run both the backend and frontend simultaneously in two separate terminals.

-   **Terminal 1: Start the Backend API**
    ```bash
    uvicorn backend_api:app --reload
    ```
    Wait for the message indicating that the application startup is complete. The API will be available at `http://localhost:8000`.

-   **Terminal 2: Start the Frontend UI**
    ```bash
    streamlit run chatbot_interface.py
    ```
    A new browser tab will open with the Streamlit chat interface. You can now ask questions.

---

## ☁️ Deployment

Deploying this application to the cloud involves two main steps:

### Step 1: Deploy the Backend to Hugging Face Spaces

The backend is deployed as a Docker image on Hugging Face Spaces.

1.  **ตรวจสอบ FAISS Index:**
    *   ตรวจสอบให้แน่ใจว่าไดเรกทอรี `faiss_index` (ซึ่งมีไฟล์ `index.faiss` และ `index.pkl`) อยู่ในไดเรกทอรีโปรเจกต์ของคุณ
    *   หากยังไม่มี คุณจะต้องสร้าง FAISS index ก่อนที่จะ build Docker image

2.  **Build Docker Image:**
    *   เปิด Terminal หรือ Command Prompt ในไดเรกทอรีโปรเจกต์ของคุณ
    *   รันคำสั่งนี้เพื่อสร้าง Docker image:
        ```bash
        docker build -t your-huggingface-username/your-space-name .
        ```
        *   แทนที่ `your-huggingface-username` ด้วยชื่อผู้ใช้ Hugging Face ของคุณ
        *   แทนที่ `your-space-name` ด้วยชื่อที่คุณต้องการสำหรับ Space ของคุณ (เช่น `my-llm-backend`)
        *   อย่าลืมจุด `.` ที่ท้ายคำสั่ง ซึ่งหมายถึง Dockerfile อยู่ในไดเรกทอรีปัจจุบัน

3.  **Log in เข้าสู่ Hugging Face CLI:**
    *   หากคุณยังไม่ได้ติดตั้ง `huggingface_hub` ให้ติดตั้งก่อน: `pip install huggingface_hub`
    *   รันคำสั่งนี้เพื่อเข้าสู่ระบบ Hugging Face:
        ```bash
        huggingface-cli login
        ```
    *   ระบบจะขอ Hugging Face Token ของคุณ คุณสามารถสร้างได้จากหน้า Settings -> Access Tokens บนเว็บไซต์ Hugging Face

4.  **Push Docker Image ไปยัง Hugging Face Spaces:**
    *   หลังจาก build image และ login แล้ว ให้รันคำสั่งนี้เพื่อ push image ไปยัง Hugging Face Container Registry:
        ```bash
        docker push your-huggingface-username/your-space-name
        ```
        *   ใช้ชื่อ image เดียวกันกับที่คุณใช้ในขั้นตอนที่ 2

5.  **สร้าง Space ใหม่บน Hugging Face:**
    *   ไปที่เว็บไซต์ Hugging Face (huggingface.co)
    *   คลิกที่โปรไฟล์ของคุณ แล้วเลือก "New Space"
    *   ตั้งชื่อ Space ของคุณ (ควรตรงกับ `your-space-name` ที่คุณใช้ใน Docker image)
    *   เลือก "Docker" เป็น SDK
    *   ในส่วน "Docker image", ระบุชื่อ Docker image ที่คุณ push ไป (เช่น `your-huggingface-username/your-space-name`)
    *   เลือกประเภทฮาร์ดแวร์ (Hardware) ที่เหมาะสม หาก LLM ของคุณต้องการ GPU ให้เลือก GPU ที่มี
    *   คลิก "Create Space"

### Step 2: Deploy the Frontend to Streamlit Cloud

1.  **Push to GitHub:** Ensure your latest code, especially `chatbot_interface.py`, is pushed to your GitHub repository.
2.  **Deploy on Streamlit:** Connect your GitHub repository to Streamlit Cloud and deploy the `chatbot_interface.py` application.
3.  **Set the Secret:** In your Streamlit app's settings, go to **Settings > Secrets** and add a new secret:
    -   **Name:** `BACKEND_API_URL`
    -   **Value:** Paste the public URL of your Hugging Face Space (e.g., `https://your-huggingface-username-your-space-name.hf.space`).

Once the secret is saved, the Streamlit app will automatically restart and will now send requests to your high-performance backend on Hugging Face Spaces, resulting in a much faster and more stable experience.