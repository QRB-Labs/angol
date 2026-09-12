#!/bin/bash
echo "Starting Enterprise Brain Nightly Cycle..."

# 1. Kill Serving LLM (Qwen) and FastAPI Middleware
pkill -f "vllm.*Qwen"
pkill -f "uvicorn serving.serve_api:app"
sleep 15 # Allow VRAM to clear completely

# 2. Start Ingestion LLM (Llama-8B) for RAPTOR Summarization
echo "Loading Meta-Llama-3.1-8B-Instruct into VRAM..."
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8000 --max-model-len 8192 &
VLLM_PID=$!
sleep 45 # Wait for model weights to load

# 3. Run the heavy RAPTOR Ingestion Pipeline
echo "Running RAPTOR Clustering & Parsing..."
python ingestion/ingestion_pipeline.py

# 4. Unload Ingestion LLM from VRAM
kill $VLLM_PID
sleep 15 

# 5. Start Serving LLM (Qwen-32B) for User Queries
echo "Loading Qwen-2.5-32B-Instruct into VRAM..."
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-32B-Instruct \
    --port 8000 --max-model-len 16384 &
sleep 60

# 6. Start FastAPI Middleware
echo "Starting LlamaIndex Routing API..."
uvicorn serving.serve_api:app --host 0.0.0.0 --port 8081 &

echo "System ready for daily user queries!"
