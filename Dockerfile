FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
       python3-pip git build-essential python3-dev cmake ninja-build pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies and GPU stack pinned to CUDA 12.4
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip setuptools wheel && \
python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 && \
    python3 -m pip install --upgrade -r /requirements.txt && \
    python3 -m pip install vllm==0.11.0 && \
(python3 -m pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.6 || echo "FlashInfer not available for this architecture, continuing without it") && \
    python3 -m pip install kvcached --no-build-isolation

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    ENABLE_KVCACHED=true \
    KVCACHED_AUTOPATCH=1 \
    KVCACHED_IPC_NAME=VLLM

ENV PYTHONPATH="/:/vllm-workspace"


COPY src /src
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
    export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
    python3 /src/download_model.py; \
    fi

# Start the handler
CMD ["python3", "/src/handler.py"]
