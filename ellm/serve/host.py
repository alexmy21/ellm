"""A second LLM in the network — its own side-car, gate, H(t), and door.

Phase 6: the host is a native-stub LLM (deterministic generator) running the
same travel-loop shape as the eLLM. Its side-car is a separate process with
its own codebook gate and its own `H(t)`; its `door.py` is the same `Door`
class the eLLM uses. The eLLM never sees this object — it only sees the
door's verdicts over the broadcast.
"""

from __future__ import annotations

from ..lattice.client import LatticeClient
from ..travel.door import Door
from .generator import Generator, StubGenerator


class HostLLM:
    """A non-eLLM host: own ewm-sm + `hllsetLUT` + `H(t)` + `door.py`."""

    def __init__(
        self,
        name: str,
        client: LatticeClient,
        tokens: list[str] | None = None,
        generator: Generator | None = None,
    ):
        self.name = name
        self.client = client
        self.generator = generator or StubGenerator()
        if tokens:
            client.gate_put(name, tokens)
        self.door = Door(client, gate=name)

    def turn(self, ids: list[int]) -> dict:
        """One host turn: ingest → materialize-to-prompt → generate."""
        self.client.ingest(ids)
        data = self.client.to_prompt(ids, user=self.name)
        context_tokens = data.get("tokens", [])
        generated = self.generator.generate(context_tokens)
        return {
            "name": self.name,
            "turn": int(self.client.state().get("turn", 0)),
            "context_tokens": context_tokens,
            "generated_ids": generated,
        }
