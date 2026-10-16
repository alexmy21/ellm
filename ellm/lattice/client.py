"""Subprocess client for the `ellm-lattice` JSON-lines side-car."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any


class LatticeError(RuntimeError):
    """Raised when the lattice side-car returns an error or dies."""


class LatticeClient:
    """A subprocess client for the `ellm-lattice` Rust binary."""

    def __init__(self, binary: str | None = None, repo_path: str | None = None):
        self.binary = binary or self._find_binary()
        cmd = [self.binary]
        if repo_path is not None:
            cmd.append(repo_path)
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        self._next_id = 0

    @staticmethod
    def _find_binary() -> str:
        """Locate the dev-built binary under `lattice/target/{debug,release}`."""
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        for profile in ("debug", "release"):
            path = os.path.join(
                repo_root, "lattice", "target", profile, "ellm-lattice"
            )
            if os.path.exists(path):
                return path
        raise LatticeError(
            "ellm-lattice binary not found; run `cargo build` in lattice/"
        )

    def _call(self, cmd: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send one request and read one response."""
        if self._proc.stdin is None or self._proc.stdout is None:
            raise LatticeError("lattice process is not running")

        self._next_id += 1
        request = {"id": str(self._next_id), "cmd": cmd, "args": args or {}}
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()

        line = self._proc.stdout.readline()
        if not line:
            raise LatticeError("lattice process closed unexpectedly")
        response = json.loads(line)
        if not response.get("ok"):
            raise LatticeError(response.get("error", "unknown lattice error"))
        return response.get("data", {})

    def ping(self) -> dict[str, Any]:
        """Health check: returns `{"pong": true, "version": …}`."""
        return self._call("ping")

    def ingest(self, ids: list[int]) -> dict[str, Any]:
        """One [UM] turn: ingest → join → ring-push → commit → door round-trip."""
        return self._call("ingest", {"ids": ids})

    def materialize(self, ids: list[int]) -> dict[str, Any]:
        """Pure door demo over `tid{n}` ids — no commit."""
        return self._call("materialize", {"ids": ids})

    def state(self) -> dict[str, Any]:
        """The live `LatticeState` snapshot (head, working set, D/R/N, ring)."""
        return self._call("state")

    def commit(self, message: str | None = None) -> dict[str, Any]:
        """Force-commit the current working set."""
        return self._call("commit", {"message": message})

    def hllset_lut(self) -> list[dict[str, Any]]:
        """The per-ewm-sm HLLSet registry (gate + confirmation), ranked by TH."""
        data = self._call("hllset_lut")
        return data["entries"]

    def gate_put(self, user: str, tokens: list[str]) -> dict[str, Any]:
        """Register a per-host codebook gate (host encodings as languages)."""
        return self._call("gate", {"op": "put", "user": user, "tokens": tokens})

    def gate_list(self) -> list[str]:
        """The registered gate names."""
        data = self._call("gate", {"op": "list"})
        return data["gates"]

    def gate_get(self, user: str) -> dict[str, Any]:
        """Read one gate catalog: `{catalog_key, tokens}`."""
        return self._call("gate", {"op": "get", "user": user})

    def to_prompt(self, ids: list[int], user: str | None = None) -> dict[str, Any]:
        """Materialize-to-prompt through the named codebook gate (the door).

        With `user` set, the door is guarded by the round-trip coverage
        invariant: `coverage = 1.0` or the response carries `refused: true`.
        """
        args: dict[str, Any] = {"ids": ids}
        if user is not None:
            args["user"] = user
        return self._call("to_prompt", args)

    def door_probe(self, hllset_hex: str, user: str | None = None) -> list[dict[str, Any]]:
        """The door guard's verdict on a broadcast HLLSet candidate.

        Returns `[{user, coverage, can_restore}, …]` — one verdict for the
        named gate, or every registered gate when `user` is omitted.
        """
        args: dict[str, Any] = {"hllset": hllset_hex}
        if user is not None:
            args["user"] = user
        data = self._call("door_probe", args)
        return data["verdicts"]

    def policy_put(self, policy: dict[str, Any]) -> dict[str, Any]:
        """Publish the active policy (the trainer's commit-as-publication)."""
        return self._call("policy_put", {"policy": policy})

    def policy_get(self) -> dict[str, Any] | None:
        """Read the active policy (the server's hot-swap read)."""
        return self._call("policy_get")["policy"]

    def setkey_hex(self, query_hex: str, nodes_hex: list[str]) -> list[float]:
        """Mode B scores over raw HLLSet bytes (discovery scorer)."""
        data = self._call("setkey_hex", {"query": query_hex, "nodes": nodes_hex})
        return data["scores"]

    def setkey(self, query: list[str], nodes: list[list[str]]) -> list[float]:
        """Mode B scores of one query HLLSet against candidate node HLLSets."""
        data = self._call("setkey", {"query": query, "nodes": nodes})
        return data["scores"]

    def close(self) -> None:
        """Shut the side-car process down."""
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        self._proc.wait(timeout=5)

    def __enter__(self) -> "LatticeClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
