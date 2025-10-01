# server.py
import streamlit.web.bootstrap
from fastapi import FastAPI
import uvicorn
import os
import subprocess
import threading

app = FastAPI()

# Start Streamlit in a separate thread to avoid blocking FastAPI
def run_streamlit():
    # Set Streamlit to run your app.py from the src/ folder
    os.environ["STREAMLIT_SERVER_PORT"] = "8000"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "True"
    subprocess.run(
        ["streamlit", "run", "src/app.py", "--server.port=8000", "--server.headless=True"],
        check=True
    )

# Start Streamlit when the server starts
@app.on_event("startup")
async def startup_event():
    thread = threading.Thread(target=run_streamlit, daemon=True)
    thread.start()

# Health check endpoint for Vercel
@app.get("/health")
async def health():
    return {"status": "ok"}

# Root endpoint to redirect to Streamlit
@app.get("/{path:path}")
async def serve_streamlit(path: str):
    return {"message": "Streamlit is running at this URL"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)