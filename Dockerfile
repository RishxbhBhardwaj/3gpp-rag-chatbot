FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install CPU-only torch first to save space
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install rest of dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Download specs and build vector store during build
RUN python download_specs.py && python ingest.py --force

# Expose port 7860 (HuggingFace Spaces default)
EXPOSE 7860

# Start FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
