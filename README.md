
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
    - Designed to be containerized with Docker and deployed on a cloud service like AWS (SageMaker, ECS, etc.).

2.  **Frontend (Streamlit):**
    - A lightweight Python application built with Streamlit.
    - Provides a simple, clean chat interface for the user.
    - Does **not** load any models or perform any AI processing.
    - When a user asks a question, it makes an HTTP request to the Backend API, receives the answer, and displays it.
    - Designed to be deployed on Streamlit Community Cloud.

### Flow Diagram

```
[User] <--> [Streamlit Frontend (UI)] <--> [FastAPI Backend (API on AWS)] <--> [LLM & Vector Store]
```

---

## 🚀 Getting Started

### 1. Initial Setup

First, you need to prepare the knowledge base and install dependencies.

- **Clone the repository:**
  ```bash
  git clone <repository_url>
  cd 04-chatbot_RAG_LLM
  ```

- **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```

- **Create the Vector Store:**
  Make sure the `agnos_forum_data.json` file is in the root directory. Then, run the `rag_pipeline.py` script **once** to create the `faiss_index` directory.
  ```bash
  python rag_pipeline.py
  ```

### 2. Running Locally for Testing

To test the full application on your local machine, you need to run both the backend and frontend simultaneously in two separate terminals.

- **Terminal 1: Start the Backend API**
  ```bash
  uvicorn backend_api:app --reload
  ```
  Wait for the message indicating that the application startup is complete. The API will be available at `http://localhost:8000`.

- **Terminal 2: Start the Frontend UI**
  ```bash
  streamlit run chatbot_interface.py
  ```
  A new browser tab will open with the Streamlit chat interface. You can now ask questions.

---

## ☁️ Deployment

Deploying this application to the cloud involves two main steps:

### Step 1: Deploy the Backend to AWS

1.  **Containerize:** Package the FastAPI application (`backend_api.py`) along with the `faiss_index` directory, `agnos_forum_data.json`, and other necessary files into a Docker container.
2.  **Choose a Service:** Push the container image to a registry (like Amazon ECR) and deploy it on an AWS service capable of hosting containers and serving HTTP traffic. Recommended services:
    - **Amazon SageMaker:** Ideal for ML workloads, provides serverless inference endpoints.
    - **Amazon ECS (Elastic Container Service) or EKS (Elastic Kubernetes Service):** General-purpose container orchestration.
    - **AWS Lambda:** Can be used if the container size is within the limits.
3.  **Get the URL:** Once deployed, AWS will provide you with a public URL for your API endpoint.

### Step 2: Deploy the Frontend to Streamlit Cloud

1.  **Push to GitHub:** Ensure your latest code, especially `chatbot_interface.py`, is pushed to your GitHub repository.
2.  **Deploy on Streamlit:** Connect your GitHub repository to Streamlit Cloud and deploy the `chatbot_interface.py` application.
3.  **Set the Secret:** In your Streamlit app's settings, go to **Settings > Secrets** and add a new secret:
    - **Name:** `BACKEND_API_URL`
    - **Value:** Paste the API URL you got from AWS in the previous step.

Once the secret is saved, the Streamlit app will automatically restart and will now send requests to your high-performance backend on AWS, resulting in a much faster and more stable experience.
