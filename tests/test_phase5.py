"""Phase 5 integration test: LoRA broadcast-policy on Qwen2.5-1.5B.

GPU-gated: the whole module is skipped when CUDA is unavailable. The model
is loaded once per session (module-scoped fixture).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("Phase 5 tests require CUDA", allow_module_level=True)

from ellm.lattice import LatticeClient  # noqa: E402
from ellm.model.base import load_qwen_hub  # noqa: E402
from ellm.model.generator_qwen import QwenGenerator  # noqa: E402
from ellm.model.ranker import QwenBroadcastRanker  # noqa: E402
from ellm.train.trainer import GuardSet, LoRALearner, Trainer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_binary() -> None:
    binary = REPO_ROOT / "lattice" / "target" / "debug" / "ellm-lattice"
    if binary.exists():
        return
    result = subprocess.run(
        ["cargo", "build", "--offline"],
        cwd=REPO_ROOT / "lattice",
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cargo build failed:\n{result.stderr}")


@pytest.fixture(scope="module")
def hub():
    _ensure_binary()
    return load_qwen_hub()


@pytest.fixture(scope="module")
def seeded_client():
    _ensure_binary()
    client = LatticeClient()
    client.gate_put("ellm", [f"tid{i}" for i in range(1, 32)])
    yield client
    client.close()


def test_hub_scores_batch(hub) -> None:
    logits = hub.score(["a b c", "d e f"])
    assert tuple(logits.shape) == (2, 1)


def test_ranker_orders_candidates(hub, seeded_client) -> None:
    seeded_client.ingest([1, 2])
    seeded_client.ingest([2, 3])
    ranker = QwenBroadcastRanker(hub, seeded_client)
    state = seeded_client.state()
    candidates = [state["working_hex"]]
    ordered = ranker.discover(state["working_hex"], candidates)
    assert ordered == candidates


def test_training_reduces_kl_loss(hub, seeded_client) -> None:
    from ellm.curriculum.native_labels import collect_samples

    ranker = QwenBroadcastRanker(hub, seeded_client)
    samples = collect_samples(
        seeded_client,
        [[1, 2], [2, 3], [3, 4]],
        ranker,
        gate="ellm",
        tau=0.1,
    )
    assert len(samples) >= 1

    learner = LoRALearner(ranker, tau=0.1)
    losses = learner.train(samples, epochs=2)
    assert len(losses) >= 2
    assert losses[-1] <= losses[0] + 1e-6


def test_qwen_generator_returns_tid_ids(hub) -> None:
    generator = QwenGenerator(hub, max_new_tokens=8)
    ids = generator.generate(["tid1", "tid2"])
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) > 0


def test_guard_set_regression_and_progress() -> None:
    guards = GuardSet(regression_slack=0.05, accept_min_gain=0.0)
    accepted, report = guards.evaluate(0.5, 0.6)
    assert accepted is True and "veto" not in report
    rejected, report = guards.evaluate(0.6, 0.5)
    assert rejected is False and report["veto"] == "RegressionGuard"


def test_trainer_publishes_guarded_policy(tmp_path: Path, hub) -> None:
    _ensure_binary()
    client = LatticeClient(repo_path=str(tmp_path / "ewm-repo"))
    try:
        client.gate_put("ellm", [f"tid{i}" for i in range(1, 32)])
        ranker = QwenBroadcastRanker(hub, client)
        trainer = Trainer(
            client,
            schedule_every=1,
            ranker=ranker,
            probes=[[1, 2, 3], [2, 3, 4]],
        )
        published = trainer.step(turns=[[1, 2], [2, 3], [3, 4]])
        assert published is not None
        assert "probe_before" in published and "probe_after" in published
        assert published["generation"] == 1
    finally:
        client.close()
