#!/usr/bin/env bash
set -euo pipefail

# ---------- styling ----------
GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; RESET="\033[0m"
info()  { echo -e "${GREEN}=>${RESET} $*"; }
warn()  { echo -e "${YELLOW}!!${RESET} $*"; }
error() { echo -e "${RED}xx${RESET} $*" >&2; }

LLAMA_PID=""
NGROK_PID=""
cleanup() {
    [[ -n "$LLAMA_PID" ]] && kill -0 "$LLAMA_PID" 2>/dev/null && kill "$LLAMA_PID" 2>/dev/null
    [[ -n "$NGROK_PID" ]] && kill -0 "$NGROK_PID" 2>/dev/null && kill "$NGROK_PID" 2>/dev/null
    return 0
}
trap cleanup EXIT INT TERM

echo "=== llama.cpp + ngrok Setup ==="
echo

# ---------- input ----------
read -p "Enter your ngrok authtoken: " NGROK_TOKEN
[[ -z "$NGROK_TOKEN" ]] && { error "ngrok authtoken is required."; exit 1; }

echo
read -p "Enter direct GGUF download URL: " MODEL_URL
[[ -z "$MODEL_URL" ]] && { error "Model URL is required."; exit 1; }

echo
read -p "Context size [default 4096]: " CTX_SIZE
CTX_SIZE=${CTX_SIZE:-4096}

echo
read -p "Port [default 8000]: " PORT
PORT=${PORT:-8000}

echo
read -p "Model alias for API responses [default = filename]: " MODEL_ALIAS

# ---------- packages ----------
info "Installing system packages"
apt-get update -qq
apt-get install -y -qq git wget curl cmake build-essential lsof > /dev/null

pip install -q pyngrok

# ---------- ngrok ----------
if ! command -v ngrok >/dev/null 2>&1; then
    info "Installing ngrok"
    curl -sSL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz | tar xz
    mv ngrok /usr/local/bin/
else
    info "ngrok already installed, skipping download"
fi
ngrok config add-authtoken "$NGROK_TOKEN" > /dev/null

# ---------- GPU detection ----------
info "Detecting GPU"
CMAKE_GPU_FLAGS=()
RUNTIME_GPU_FLAGS=()
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    info "NVIDIA GPU detected: $GPU_NAME"

    if ! command -v nvcc >/dev/null 2>&1 && [[ -x /usr/local/cuda/bin/nvcc ]]; then
        export PATH="/usr/local/cuda/bin:$PATH"
    fi

    if command -v nvcc >/dev/null 2>&1; then
        CMAKE_GPU_FLAGS=(-DGGML_CUDA=ON)
        RUNTIME_GPU_FLAGS=(-ngl 999 --flash-attn)
        info "CUDA toolkit found — GPU build enabled with full offload"
    else
        warn "nvidia-smi works but nvcc not found — building CPU-only (try: apt-get install -y nvidia-cuda-toolkit)"
    fi
else
    info "No NVIDIA GPU detected — building CPU-only"
fi

# ---------- clone / build ----------
if [[ -d llama.cpp ]]; then
    info "llama.cpp already present, pulling latest"
    cd llama.cpp
    git pull --ff-only
else
    info "Cloning llama.cpp"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
    cd llama.cpp
fi

info "Building llama.cpp (this can take a few minutes)"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON "${CMAKE_GPU_FLAGS[@]}"
cmake --build build -j"$(nproc)"

mkdir -p models

# ---------- download model ----------
info "Resolving model filename"
MODEL_FILENAME=$(curl -sIL "$MODEL_URL" 2>/dev/null \
    | grep -i "content-disposition" \
    | sed -n 's/.*filename="\?\([^"\r\n;]*\)"\?.*/\1/p' \
    | tail -n1 || true)
if [[ -z "$MODEL_FILENAME" ]]; then
    MODEL_FILENAME=$(basename "${MODEL_URL%%\?*}")
fi
[[ "$MODEL_FILENAME" != *.gguf ]] && MODEL_FILENAME="${MODEL_FILENAME}.gguf"
MODEL_PATH="models/$MODEL_FILENAME"

info "Downloading model -> $MODEL_PATH (resumable)"
wget -c --tries=5 --timeout=60 -O "$MODEL_PATH" "$MODEL_URL"

[[ ! -s "$MODEL_PATH" ]] && { error "Model download failed or file is empty."; exit 1; }
info "Model size: $(du -h "$MODEL_PATH" | cut -f1)"

[[ -z "$MODEL_ALIAS" ]] && MODEL_ALIAS="$MODEL_FILENAME"

# ---------- port check ----------
if lsof -i:"$PORT" -t >/dev/null 2>&1; then
    error "Port $PORT is already in use. Pick another port and re-run."
    exit 1
fi

# ---------- start server ----------
info "Starting llama-server on port $PORT (ctx=$CTX_SIZE)"
./build/bin/llama-server \
    -m "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    -c "$CTX_SIZE" \
    --alias "$MODEL_ALIAS" \
    --jinja \
    --threads "$(nproc)" \
    "${RUNTIME_GPU_FLAGS[@]}" \
    > llama.log 2>&1 &
LLAMA_PID=$!

info "Waiting for llama-server to become healthy..."
for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        info "llama-server is up"
        break
    fi
    if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
        error "llama-server exited unexpectedly. Last log lines:"
        tail -n 30 llama.log
        exit 1
    fi
    if [[ "$i" -eq 60 ]]; then
        error "llama-server did not become healthy in time. Last log lines:"
        tail -n 30 llama.log
        exit 1
    fi
    sleep 2
done

# ---------- start ngrok ----------
info "Starting ngrok tunnel"
ngrok http "$PORT" --log=stdout > ngrok.log 2>&1 &
NGROK_PID=$!

URL=""
for i in $(seq 1 20); do
    URL=$(curl -s http://127.0.0.1:4040/api/tunnels | grep -o 'https://[^"]*' | head -n1 || true)
    [[ -n "$URL" ]] && break
    sleep 1
done

if [[ -z "$URL" ]]; then
    error "Could not retrieve ngrok URL. Check ngrok.log for details."
    exit 1
fi

echo
echo "======================================"
echo "Server running!"
echo
echo "Model:        $MODEL_FILENAME (alias: $MODEL_ALIAS)"
echo "Context size: $CTX_SIZE"
echo "GPU offload:  ${RUNTIME_GPU_FLAGS[*]:-none (CPU only)}"
echo "Port:         $PORT"
echo
echo "OpenAI endpoint:"
echo "$URL"
echo
echo "Examples:"
echo "$URL/v1/chat/completions"
echo "$URL/v1/models"
echo
echo "API Key: anything"
echo "======================================"

wait "$LLAMA_PID" "$NGROK_PID"
