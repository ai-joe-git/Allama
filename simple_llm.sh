#!/bin/bash

# Simple LLM Tool Runner
# Usage: ./simple_llm.sh <model_name> [generate|chat]

set -e

MODELS_DIR="$HOME/.simple_llm_models"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON_SCRIPT="$SCRIPT_DIR/simple_llm_runner.py"
HF_ACCOUNT="mradermacher"

# Function to search for GGUF models
search_models() {
    echo "Searching for GGUF models from $HF_ACCOUNT..."
    models=$(curl -s "https://huggingface.co/$HF_ACCOUNT/api/models?sort=lastModified&direction=-1&limit=100" | grep -o '"id":"[^"]*"' | cut -d'"' -f4 | grep "GGUF")
    echo "$models" | nl
    echo "Enter the number of the model you want to use:"
    read choice
    MODEL_NAME=$(echo "$models" | sed -n "${choice}p")
}

# Function to download a model
download_model() {
    local model_name="$1"
    local model_url="https://huggingface.co/$HF_ACCOUNT/$model_name/resolve/main/$model_name.gguf"
    local model_path="$MODELS_DIR/$model_name.gguf"

    if [ ! -f "$model_path" ]; then
        echo "Downloading $model_name..."
        mkdir -p "$MODELS_DIR"
        curl -L "$model_url" -o "$model_path"
        echo "Download complete."
    fi
}

# Check if a model name is provided or if we need to search
if [ $# -lt 1 ] || [ "$1" == "search" ]; then
    search_models
else
    MODEL_NAME="$1"
fi

MODE="${2:-chat}"  # Default to chat mode if not specified

# Download the model if it doesn't exist
download_model "$MODEL_NAME"

# Run the Python script
if [ "$MODE" = "generate" ]; then
    python "$PYTHON_SCRIPT" "$MODELS_DIR/$MODEL_NAME.gguf" generate
elif [ "$MODE" = "chat" ]; then
    python "$PYTHON_SCRIPT" "$MODELS_DIR/$MODEL_NAME.gguf" chat
else
    echo "Invalid mode. Use 'generate' or 'chat'."
    exit 1
fi
