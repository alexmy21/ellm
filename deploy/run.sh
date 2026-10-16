#!/usr/bin/env bash
# Run the LLM-net demo: two containers, one shared lattice volume, GPU if present.
#
#   ./deploy/run.sh
#   → open http://localhost:8080
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DEPLOY_DIR/.."
REPO_ROOT="$(pwd)"

if ! command -v podman >/dev/null 2>&1; then
    echo "error: podman is not installed" >&2
    exit 1
fi

# Compose backend. Prefer the Python `podman-compose`, which talks to podman
# directly. The native `podman compose` wrapper can delegate to the Docker CLI
# plugin and fail with "Cannot connect to the Docker daemon" when the podman
# user socket is not running — avoid that surprise.
if command -v podman-compose >/dev/null 2>&1; then
    COMPOSE=(podman-compose)
else
    COMPOSE=(podman compose)
fi

# Build the release side-car if missing (the image needs it).
if [ ! -x lattice/target/release/ellm-lattice ]; then
    echo "==> side-car binary missing — building it"
    cargo build --release --manifest-path lattice/Cargo.toml
fi

# Build the CPU image if missing.
if ! podman image exists ellm-demo:cpu 2>/dev/null; then
    ./deploy/build.sh
fi

# Absolute compose paths: podman-compose on some systems does not resolve
# relative `-f` paths against the current directory.
FILES=(-f "$DEPLOY_DIR/podman-compose.yml")
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    if ! podman image exists ellm-demo:gpu 2>/dev/null; then
        echo "==> GPU detected — building the GPU image first"
        podman build \
            --build-arg BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime \
            --build-arg ELLM_EXTRAS="transformers peft" \
            -t ellm-demo:gpu -f "$DEPLOY_DIR/Containerfile" "$REPO_ROOT"
    fi
    FILES+=(-f "$DEPLOY_DIR/podman-compose.gpu.yml")
    echo "==> running with GPU passthrough"
else
    echo "==> running CPU-only"
fi

echo "==> starting the network (Ctrl-C to stop)"
"${COMPOSE[@]}" "${FILES[@]}" up

echo
echo "demo stopped; to restart: ${COMPOSE[*]} ${FILES[*]} up"
echo "open http://localhost:8080 while it is running"
