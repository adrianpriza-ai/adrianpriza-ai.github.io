#!/usr/bin/env bash
set -euo pipefail

# ---------- styling ----------
GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; RESET="\033[0m"
info()  { echo -e "${GREEN}==>${RESET} $*"; }
warn()  { echo -e "${YELLOW}!!${RESET} $*"; }
error() { echo -e "${RED}xx${RESET} $*" >&2; }

KOBOLD_PID=""
NGROK_PID=""
cleanup() {
    [[ -n "$KOBOLD_PID" ]] && kill -0 "$KOBOLD_PID" 2>/dev/null && kill "$KOBOLD_PID" 2>/dev/null
    [[ -n "$NGROK_PID" ]] && kill -0 "$NGROK_PID" 2>/dev/null && kill "$NGROK_PID" 2>/dev/null
    return 0
}
trap cleanup EXIT INT TERM

echo "=== koboldcpp + ngrok Setup ==="
echo

# ---------- input ----------
# When this script is run via `curl ... | bash`, stdin is the script itself,
# not the keyboard. Read from /dev/tty instead so prompts actually work.
if [[ -t 0 ]]; then
    TTY="/dev/stdin"
elif [[ -r /dev/tty ]]; then
    TTY="/dev/tty"
else
    error "No interactive terminal available to read input (are you piping this into a non-interactive shell?)."
    error "Download it first and run it directly instead: wget -O setup.sh <url> && bash setup.sh"
    exit 1
fi

read -p "Enter your ngrok authtoken: " NGROK_TOKEN < "$TTY"
[[ -z "$NGROK_TOKEN" ]] && { error "ngrok authtoken is required."; exit 1; }

echo
read -p "Enter direct GGUF download URL: " MODEL_URL < "$TTY"
[[ -z "$MODEL_URL" ]] && { error "Model URL is required."; exit 1; }

echo
read -p "Context size [default 4096]: " CTX_SIZE < "$TTY"
CTX_SIZE=${CTX_SIZE:-4096}

echo
read -p "Port [default 5001]: " PORT < "$TTY"
PORT=${PORT:-5001}

echo
read -p "Model alias for API responses [default = filename]: " MODEL_ALIAS < "$TTY"

# ---------- packages ----------
info "Installing system packages"
apt-get update -qq
apt-get install -y -qq wget curl lsof > /dev/null

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
GPU_RUNTIME_FLAGS=()
HAS_GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    info "NVIDIA GPU detected: $GPU_NAME"
    HAS_GPU=1
    GPU_RUNTIME_FLAGS=(--usecublas mmq --gpulayers 999)
else
    info "No NVIDIA GPU detected — will run CPU-only build"
fi

# ---------- fetch latest koboldcpp release ----------
info "Looking up latest koboldcpp release"
RELEASE_JSON=$(curl -sSL https://api.github.com/repos/LostRuins/koboldcpp/releases/latest)
KCPP_VERSION=$(echo "$RELEASE_JSON" | grep -m1 '"tag_name"' | sed -E 's/.*"tag_name": *"([^"]+)".*/\1/')

if [[ "$HAS_GPU" -eq 1 ]]; then
    ASSET_NAME="koboldcpp-linux-x64"
else
    ASSET_NAME="koboldcpp-linux-x64-nocuda"
fi

KCPP_URL=$(echo "$RELEASE_JSON" \
    | grep "browser_download_url" \
    | grep "$ASSET_NAME\"" \
    | sed -E 's/.*"(https[^"]+)".*/\1/' \
    | head -n1)

if [[ -z "$KCPP_URL" ]]; then
    error "Could not resolve download URL for asset '$ASSET_NAME' from latest release ($KCPP_VERSION)."
    exit 1
fi

info "Latest koboldcpp: $KCPP_VERSION ($ASSET_NAME)"

# ---------- download koboldcpp binary ----------
KCPP_BIN="koboldcpp"
if [[ -f "$KCPP_BIN" && -x "$KCPP_BIN" ]]; then
    info "koboldcpp binary already present, re-downloading to ensure it's the latest"
fi

info "Downloading koboldcpp binary"
wget -q --show-progress -O "$KCPP_BIN" "$KCPP_URL"
chmod +x "$KCPP_BIN"

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

# ---------- start koboldcpp ----------
info "Starting koboldcpp on port $PORT (ctx=$CTX_SIZE)"
./"$KCPP_BIN" \
    -m "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --contextsize "$CTX_SIZE" \
    --threads "$(nproc)" \
    --skiplauncher \
    "${GPU_RUNTIME_FLAGS[@]}" \
    > koboldcpp.log 2>&1 &
KOBOLD_PID=$!

info "Waiting for koboldcpp to become healthy..."
for i in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:${PORT}/api/v1/model" >/dev/null 2>&1; then
        info "koboldcpp is up"
        break
    fi
    if ! kill -0 "$KOBOLD_PID" 2>/dev/null; then
        error "koboldcpp exited unexpectedly. Last log lines:"
        tail -n 30 koboldcpp.log
        exit 1
    fi
    if [[ "$i" -eq 90 ]]; then
        error "koboldcpp did not become healthy in time. Last log lines:"
        tail -n 30 koboldcpp.log
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
echo "koboldcpp:    $KCPP_VERSION ($ASSET_NAME)"
echo "Model:        $MODEL_FILENAME"
echo "Context size: $CTX_SIZE"
echo "GPU offload:  ${GPU_RUNTIME_FLAGS[*]:-none (CPU only)}"
echo "Port:         $PORT"
echo
echo "OpenAI-compatible endpoint:"
echo "$URL"
echo
echo "Examples:"
echo "$URL/v1/chat/completions"
echo "$URL/v1/models"
echo
echo "KoboldAI-native UI:"
echo "$URL"
echo
echo "API Key: anything"
echo "======================================"

wait "$KOBOLD_PID" "$NGROK_PID"
