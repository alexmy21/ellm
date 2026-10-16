#!/usr/bin/env python3
"""Phase 0 GPU smoke test — verify the RTX 3060 is usable from PyTorch.

Run with the ewm-nanolm env:

    /home/alexmy/.conda/envs/ewm-nanolm/bin/python scripts/smoke_gpu.py
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("FAIL: torch not installed (expected in the ewm-nanolm env)")
        return 1

    print(f"torch {torch.__version__} — cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        return 1

    name = torch.cuda.get_device_name(0)
    total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"device 0: {name} ({total_gib:.1f} GiB total)")

    if "3060" not in name:
        print(f"WARN: expected an RTX 3060, got '{name}'")

    x = torch.randn(2048, 2048, device="cuda")
    y = float((x @ x).sum().item())
    print(f"smoke matmul ok (sum={y:.2f})")

    # Reserve/check a realistic fraction of VRAM is available.
    free_gib = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"free after matmul: {free_gib:.1f} GiB")
    if free_gib < 8.0:
        print("WARN: less than 8 GiB free — check what else is on the GPU")

    print("GPU smoke test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
