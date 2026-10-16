"""The deployable demo server — browser UI + JSON API over the travel loop.

Stdlib only (`http.server` + `urllib`), so the container image stays tiny.

Roles (env `ELLM_ROLE`):

* `ellm` — the browser-facing eLLM: web UI at `/`, dispatcher that broadcasts
  to its local door and every remote door (`ELLM_REMOTE_DOORS=name=url,…`).
* `host` — a host LLM's door daemon: `/api/door_probe`, `/api/turn`, `/api/state`.

Every container runs its own `ellm-lattice` side-car over its own repo
subdirectory inside the shared volume, and every door is the same code.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..lattice.client import LatticeClient
from .broadcast import BroadcastDispatcher, NativeStubPolicy
from .doors import DoorNetwork
from .generator import StubGenerator

DEFAULT_BINARY = os.environ.get(
    "ELLM_BINARY", "/usr/local/bin/ellm-lattice"
)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


class RemoteDoor:
    """A door daemon in another container, reached over HTTP."""

    def __init__(self, name: str, url: str):
        self.name = name
        self.url = url.rstrip("/")

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())

    def door_probe(self, hllset_hex: str, user: str | None = None) -> list[dict]:
        return self._post("/api/door_probe", {"hllset": hllset_hex})["verdicts"]

    def turn(self, ids: list[int]) -> dict:
        return self._post("/api/turn", {"ids": ids})

    def close(self) -> None:  # no-op: nothing to release over HTTP
        pass


class DemoRuntime:
    """One container's demo runtime: side-car + doors + generator."""

    def __init__(self) -> None:
        self.role = os.environ.get("ELLM_ROLE", "ellm")
        self.name = os.environ.get("ELLM_NAME", self.role)
        self.repo = os.environ.get("ELLM_REPO", f"/tmp/ellm-store/{self.name}")
        gate_range = os.environ.get(
            "ELLM_GATE_RANGE",
            "1,16" if self.role == "ellm" else "17,32",
        )
        lo, hi = (int(x) for x in gate_range.split(",", 1))
        self.gate_tokens = [f"tid{i}" for i in range(lo, hi + 1)]
        self.binary = os.environ.get("ELLM_BINARY", DEFAULT_BINARY)

        self.client = LatticeClient(binary=self.binary, repo_path=self.repo)
        self.client.gate_put(self.name, self.gate_tokens)

        self.generator = StubGenerator()
        self.remote_doors: dict[str, RemoteDoor] = {}
        if self.role == "ellm":
            for entry in os.environ.get("ELLM_REMOTE_DOORS", "").split(","):
                entry = entry.strip()
                if not entry:
                    continue
                name, url = entry.split("=", 1)
                self.remote_doors[name] = RemoteDoor(name, url)
            if os.environ.get("ELLM_USE_GPU", "0") == "1":
                self._attach_qwen_generator()

        self.network = DoorNetwork()
        self.network.attach(self.name, self.client)
        for name, door in self.remote_doors.items():
            self.network.attach(name, door)  # type: ignore[arg-type]
        self.dispatcher = BroadcastDispatcher(
            self.client, NativeStubPolicy(self.client), network=self.network
        )

    def _attach_qwen_generator(self) -> None:
        try:
            import torch  # noqa: F401
            from .generator_qwen import QwenGenerator
            from ..model.base import load_qwen_hub

            if torch.cuda.is_available():
                hub = load_qwen_hub()
                self.generator = QwenGenerator(hub)
        except Exception as exc:  # GPU optional: fall back to the stub
            print(f"[ellm] GPU generator unavailable ({exc}); using stub")

    # ── turn logic ──────────────────────────────────────────────────────

    def run_turn(self, ids: list[int]) -> dict:
        ingest_response = self.client.ingest(ids)
        candidate = ingest_response["turn_hex"]
        verdicts = self.dispatcher.broadcast(candidate)
        selection = self.dispatcher.select(verdicts)

        if selection.target and selection.target != self.name:
            remote = self.remote_doors.get(selection.target)
            if remote is not None:
                routed = remote.turn(ids)
                generated = routed.get("generated_ids", [])
                context_tokens = routed.get("context_tokens", [])
            else:
                generated, context_tokens = [], []
        elif selection.target == self.name:
            context_tokens = self.client.to_prompt(ids, user=self.name).get("tokens", [])
            generated = self.generator.generate(context_tokens)
        else:
            context_tokens, generated = [], []

        return {
            "role": self.role,
            "name": self.name,
            "ids": ids,
            "turn": int(self.client.state().get("turn", 0)),
            "candidate": candidate[:16] + "…",
            "verdicts": [
                {"user": v.user, "coverage": round(v.coverage, 4), "can_restore": v.can_restore}
                for v in verdicts
            ],
            "selection": {
                "target": selection.target,
                "fallback": selection.fallback,
                "needs_user": list(selection.needs_user),
            },
            "context_tokens": context_tokens,
            "generated_ids": generated,
        }

    def door_probe(self, hllset_hex: str) -> list[dict]:
        return [
            {"user": v.user, "coverage": v.coverage, "can_restore": v.can_restore}
            for v in self.dispatcher.broadcast(hllset_hex)
        ]

    def state(self) -> dict:
        return {
            "role": self.role,
            "name": self.name,
            "turn": int(self.client.state().get("turn", 0)),
            "gates": self.client.gate_list(),
            "doors": [self.name, *self.remote_doors.keys()],
            "generator": type(self.generator).__name__,
        }


def _page() -> str:
    return """<!doctype html>
<html><head><title>eLLM — LLM-net demo</title><meta charset="utf-8">
<style>
 body{font-family:system-ui,sans-serif;margin:2rem;max-width:52rem}
 table{border-collapse:collapse;width:100%} td,th{border:1px solid #ccc;padding:.4rem}
 pre{background:#f5f5f5;padding:.75rem;overflow:auto}
 button{font-size:1rem;padding:.4rem 1rem}
</style></head><body>
<h1>eLLM — the LLM-net of two</h1>
<p>The recruiter broadcasts each turn's HLLSet to every door; the doors answer
whether their codebook covers it (<code>coverage = 1.0</code>). One yes → hire,
several yeses → the CEO (you) chooses.</p>
<p>Turn ids: <input id="ids" value="1,2" size="24"/>
<button onclick="turn()">run turn</button></p>
<pre id="out">(run a turn…)</pre>
<script>
async function turn(){
  const ids=document.getElementById('ids').value.split(',').map(s=>parseInt(s.trim(),10));
  document.getElementById('out').textContent='…';
  const r=await fetch('/api/turn',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ids})});
  document.getElementById('out').textContent=JSON.stringify(await r.json(),null,2);
}
fetch('/api/state').then(r=>r.json()).then(s=>{
  document.title='eLLM — '+s.name;
  document.querySelector('h1').textContent='eLLM — '+s.name+' ('+s.doors.join(' + ')+')';
});
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    runtime: DemoRuntime

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            body = _page().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            self._json(self.runtime.state())
        elif self.path == "/api/ping":
            self._json({"name": self.runtime.name, "role": self.runtime.role})
        else:
            self._json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad json"}, status=400)

        if self.path == "/api/turn":
            self._json(self.runtime.run_turn(payload.get("ids", [])))
        elif self.path == "/api/door_probe":
            self._json({"verdicts": self.runtime.door_probe(payload.get("hllset", ""))})
        else:
            self._json({"error": "not found"}, status=404)

    def log_message(self, fmt: str, *args) -> None:  # quieter container logs
        print(f"[http] {self.address_string()} {fmt % args}")


def serve(port: int) -> ThreadingHTTPServer:
    runtime = DemoRuntime()
    handler = type("Handler", (_Handler,), {"runtime": runtime})
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    print(f"[ellm-http] {runtime.name} ({runtime.role}) on :{port} "
          f"repo={runtime.repo} doors={runtime.state()['doors']}")
    return server


def main() -> None:
    port = _int_env("ELLM_PORT", 8080 if os.environ.get("ELLM_ROLE", "ellm") == "ellm" else 8001)
    server = serve(port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
