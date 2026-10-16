#!/usr/bin/env bash
# Build the demo images.
#
#   ./deploy/build.sh          # CPU image always; GPU image if the host has it
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DEPLOY_DIR/.."
REPO_ROOT="$(pwd)"

echo "==> building the Rust side-car (release)"
cargo build --release --manifest-path lattice/Cargo.toml

echo "==> building ellm-demo:cpu"
podman build -t ellm-demo:cpu -f "$DEPLOY_DIR/Containerfile" "$REPO_ROOT"

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "==> NVIDIA GPU detected — building ellm-demo:gpu"
    podman build \
        --build-arg BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime \
        --build-arg ELLM_EXTRAS="transformers peft" \
        -t ellm-demo:gpu -f "$DEPLOY_DIR/Containerfile" "$REPO_ROOT"
else
    echo "==> no NVIDIA GPU — skipping the GPU image"
fi

echo "==> done: ellm-demo:cpu ready"
