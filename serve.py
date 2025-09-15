import subprocess
import sys
import os
import logging
# import boto3 # No longer needed for S3 download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the local path for the FAISS index
# For Hugging Face Spaces, we'll assume the index is copied into the image
FAISS_INDEX_LOCAL_PATH = os.path.join(os.getcwd(), "faiss_index") # Assuming faiss_index is at the root of the app

# Removed S3 download logic as we are pre-packaging the FAISS index for Hugging Face Spaces
# def download_faiss_index(s3_path, local_path):
#     ... (removed S3 download function)

def start_server():
    """Starts the Uvicorn server."""
    logger.info("Starting Uvicorn server...")
    # Set the FAISS_INDEX_PATH environment variable for the FastAPI app
    os.environ["FAISS_INDEX_PATH"] = FAISS_INDEX_LOCAL_PATH
    
    # Get port from environment variable, default to 8080
    port = int(os.environ.get("PORT", 8080))
    
    # Command to run Uvicorn
    cmd = ["uvicorn", "backend_api:app", "--host", "0.0.0.0", "--port", str(port)]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    # No S3 download needed, FAISS index is expected to be in the image
    # download_faiss_index(FAISS_S3_PATH, FAISS_INDEX_LOCAL_PATH)
    start_server()