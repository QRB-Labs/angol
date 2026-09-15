#!/bin/bash
echo "Starting Enterprise Brain Nightly Cycle..."

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
pkill -f "uvicorn serving.serve_api:app"
sleep 10 # Allow VRAM to clear completely

# 2. Start Ingestion LLM (Llama-8B) for RAPTOR Summarization
echo "Loading Meta-Llama-3.1-8B-Instruct into VRAM via Docker..."
docker run -d --name vllm-model --gpus all \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --max-model-len 8192

wait_for_vllm

# 3. Run the heavy RAPTOR Ingestion Pipeline
echo "Running RAPTOR Clustering & Parsing..."
python ingestion/ingestion_pipeline.py

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
uvicorn serving.serve_api:app --host 0.0.0.0 --port 8081 &

echo "System ready for daily user queries!"
