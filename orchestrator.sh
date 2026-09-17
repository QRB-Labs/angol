#!/bin/bash
echo "Starting Angol Cycle..."

# Ensure we are in the project root directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR" || exit 1

# Define absolute paths to virtual environment executables
VENV_PYTHON="${PROJECT_DIR}/angol_env/bin/python"
VENV_UVICORN="${PROJECT_DIR}/angol_env/bin/uvicorn"

wait_for_vllm() {
    echo "Waiting for vLLM to be ready on port 8000..."
    while ! curl -s http://localhost:8000/v1/models > /dev/null; do
	sleep 5
    done
    echo "vLLM is up and ready!"
}

# 1. Kill Serving LLM and FastAPI Middleware
echo "Shutting down Serving Model and Middleware..."
docker stop vllm-model 2>/dev/null
docker rm vllm-model 2>/dev/null
pkill -f "serving.serve_api:app"
sleep 10 # Allow VRAM to clear completely

# 2. Start Ingestion LLM (Llama-8B) for RAPTOR Summarization
echo "Loading Meta-Llama-3.1-8B-Instruct into VRAM via Docker..."
docker run -d --name vllm-model --gpus all \
    -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}" \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85

wait_for_vllm

# 3. Run the heavy RAPTOR Ingestion Pipeline
echo "Running RAPTOR Clustering & Parsing..."
$VENV_PYTHON ingestion/ingestion_pipeline.py

# 4. Unload Ingestion LLM from VRAM
echo "Unloading Ingestion Model..."
docker stop vllm-model
docker rm vllm-model
sleep 10

# 5. Start Serving LLM (Qwen-32B-AWQ) for User Queries
echo "Loading Qwen-2.5-32B-Instruct-AWQ into VRAM via Docker..."
docker run -d --name vllm-model --gpus all \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-32B-Instruct-AWQ \
    --max-model-len 16384

wait_for_vllm

# 6. Start FastAPI Middleware
echo "Starting LlamaIndex Routing API..."
$VENV_UVICORN serving.serve_api:app --host 0.0.0.0 --port 8081 &

echo "System ready for daily user queries!"
