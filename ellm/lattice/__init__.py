"""Lattice side-car client — JSON lines over stdin/stdout.

The Rust binary `ellm-lattice` speaks the protocol defined in
`lattice/crates/ellm-lattice/src/main.rs`: one JSON request per line, one
JSON response per line.
"""

from .client import LatticeClient, LatticeError

__all__ = ["LatticeClient", "LatticeError"]
