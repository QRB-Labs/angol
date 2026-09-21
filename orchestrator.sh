#!/bin/bash
echo "Starting Angol orchestrator..."

# Ensure we are in the project root directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR" || exit 1

# Load .env variables into the bash environment
if [ -f "${PROJECT_DIR}/.env" ]; then
    set -a
    source "${PROJECT_DIR}/.env"
    set +a
fi

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

RAW_DIR="${PROJECT_DIR}/data/raw_documents/"

# ==========================================
# PHASE 1: CONDITIONAL INGESTION PIPELINE
# ==========================================

# Check if there are any files in the raw_documents directory
if [ -n "$(find "$RAW_DIR" -type f | head -1)" ]; then
    echo "New documents detected in ${RAW_DIR}. Initiating Ingestion Pipeline..."

    # 1. Kill Serving LLM and FastAPI Middleware
    echo "Shutting down Generation Model and Middleware..."
    docker stop vllm-model 2>/dev/null
    docker rm vllm-model 2>/dev/null
    pkill -f "serving.serve_api:app"
    sleep 10 # Allow VRAM to clear completely

    # 2. Start Ingestion LLM for RAPTOR Summarization
    echo "Loading ${INGESTION_MODEL} into VRAM via Docker..."
    docker run -d --name vllm-model --gpus all \
	--env-file "${PROJECT_DIR}/.env" \
	-p 8000:8000 \
	-v ~/.cache/huggingface:/root/.cache/huggingface \
	--ipc=host \
	vllm/vllm-openai:latest \
	--model ${INGESTION_MODEL} \
	--max-model-len 8192 \
	--gpu-memory-utilization 0.50

    wait_for_vllm

    # 3. Run the RAPTOR Ingestion Pipeline
    echo "Running RAPTOR Clustering & Parsing..."
    $VENV_PYTHON ingestion/ingestion_pipeline.py --raw-dir "$RAW_DIR" --processed-dir "${PROJECT_DIR}/data/processed_documents/"

    # 4. Unload Ingestion LLM from VRAM
    echo "Unloading Ingestion Model..."
    docker stop vllm-model
    docker rm vllm-model
    sleep 10
else
    echo "No new files found in ${RAW_DIR}. Skipping ingestion pipeline."
fi

# ==========================================
# PHASE 2: CONDITIONAL GENERATION STARTUP
# ==========================================

# 5. Start generation LLM (if not already running)
# We hit the API to see if it is responding AND if the currently loaded model is the GENERATION_MODEL
if curl -s http://localhost:8000/v1/models | grep -q "${GENERATION_MODEL}"; then
    echo "Generation Model (${GENERATION_MODEL}) is already loaded and running."
else
    echo "Generation Model is not running or incorrect model is loaded. Starting..."

    # Ensure any stuck/incorrect containers are killed before starting
    docker stop vllm-model 2>/dev/null
    docker rm vllm-model 2>/dev/null
    sleep 5

    echo "Loading ${GENERATION_MODEL} into VRAM via Docker..."
    docker run -d --name vllm-model --gpus all \
	--env-file "${PROJECT_DIR}/.env" \
	-p 8000:8000 \
	-v ~/.cache/huggingface:/root/.cache/huggingface \
	--ipc=host \
	vllm/vllm-openai:latest \
	--model ${GENERATION_MODEL} \
	--max-model-len 16384 \
	--gpu-memory-utilization 0.90

    wait_for_vllm
fi

# 6. Start FastAPI Middleware (if not already running)
if pgrep -f "serving.serve_api:app" > /dev/null; then
    echo "LlamaIndex Routing API is already running."
else
    echo "Starting LlamaIndex Routing API..."
    $VENV_UVICORN serving.serve_api:app --host 0.0.0.0 --port 8081 &
fi

echo "System ready for daily user queries!"
