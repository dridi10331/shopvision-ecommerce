import os
import subprocess
import sys

# Entry point for Hugging Face Spaces
if __name__ == "__main__":
    # Launch Streamlit app on port 7860 (required by HF Spaces)
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "frontend/app.py",
        "--server.port=7860",
        "--server.address=0.0.0.0"
    ])
