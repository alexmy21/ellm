"""The trainer process — owns the scheduler, learns WIP, commits at slow time.

Phase 5: the learner is real — LoRA soft-label KL against native ring-basis
labels on Qwen2.5-1.5B — and the guard set is real (probe regression slack +
progress gain). The contract is unchanged: the trainer publishes a policy
only after the guards pass, and the server sees it only after publication.
"""

from __future__ import annotations

from typing import Any

import torch

from ..curriculum.native_labels import collect_samples, probe_accuracy
from ..curriculum.probes import builtin_probes
from ..lattice.client import LatticeClient


class GuardSet:
    """The acceptance contract for a candidate policy."""

    def __init__(self, regression_slack: float = 0.05, accept_min_gain: float = 0.0):
        self.regression_slack = regression_slack
        self.accept_min_gain = accept_min_gain

    def evaluate(self, probe_before: float, probe_after: float) -> tuple[bool, dict[str, Any]]:
        """RegressionGuard + ProgressGuard over held-out probe accuracy."""
        report = {
            "probe_before": round(probe_before, 4),
            "probe_after": round(probe_after, 4),
        }
        if probe_after < probe_before - self.regression_slack:
            report["veto"] = "RegressionGuard"
            return False, report
        if probe_after - probe_before < self.accept_min_gain:
            report["veto"] = "ProgressGuard"
            return False, report
        return True, report


class LoRALearner:
    """Trains the LoRA adapter + score head by soft-label KL."""

    def __init__(self, ranker, lr: float = 1e-4, tau: float = 0.1):
        self.ranker = ranker
        self.tau = tau
        head_params = list(ranker.hub.score_head.parameters())
        lora_params = [p for p in ranker.hub.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            [{"params": head_params, "lr": lr}, {"params": lora_params, "lr": lr * 0.1}]
        )

    def _kl_loss(self, sample) -> torch.Tensor:
        prompts = sample.prompts(self.ranker.prompt)
        logits = self.ranker.hub.score_train(prompts).squeeze(-1).float()  # (n,) fp32
        labels = torch.tensor(
            sample.native_scores,
            device=logits.device,
            dtype=torch.float32,
        )
        tau = self.tau
        log_preds = torch.log_softmax(logits / tau, dim=-1)
        targets = torch.softmax(labels / tau, dim=-1)
        return (targets * (torch.log(targets + 1e-8) - log_preds)).sum(dim=-1).mean()

    def train(self, samples, epochs: int = 2) -> list[float]:
        """Run a few epochs of soft-label KL; returns per-step losses."""
        losses: list[float] = []
        self.ranker.hub.score_head.train()
        self.ranker.hub.model.train()
        try:
            for _ in range(epochs):
                for sample in samples:
                    if len(sample.native_scores) < 2:
                        continue
                    self.optimizer.zero_grad()
                    loss = self._kl_loss(sample)
                    if torch.isfinite(loss):
                        loss.backward()
                        trainable = [
                            p
                            for group in self.optimizer.param_groups
                            for p in group["params"]
                            if p.requires_grad
                        ]
                        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                        self.optimizer.step()
                    losses.append(float(loss.item()))
        finally:
            self.ranker.hub.score_head.eval()
            self.ranker.hub.model.eval()
        return losses


class Trainer:
    """The trainer process. Publishes the policy at slow time."""

    def __init__(
        self,
        client: LatticeClient,
        schedule_every: int = 2,
        ranker=None,
        probes: list[list[int]] | None = None,
        gate: str = "ellm",
        tau: float = 0.1,
    ):
        self.client = client
        self.schedule_every = schedule_every
        self.guards = GuardSet()
        self.ranker = ranker
        self.probes = probes if probes is not None else [list(p.ids) for p in builtin_probes()]
        self.gate = gate
        self.tau = tau
        self.generation = 0
        self.observations = 0

    def _candidate(self, current: dict[str, Any] | None, report: dict[str, Any]) -> dict[str, Any]:
        theta = float((current or {}).get("theta", 0.3))
        return {
            "generation": self.generation + 1,
            "theta": theta,
            "tau": self.tau,
            "lambda": float((current or {}).get("lambda", 0.0)),
            "door": "surprise-v0",
            **report,
        }

    def step(self, turns: list[list[int]] | None = None) -> dict[str, Any] | None:
        """One slow-time tick: learn WIP, run guards, publish or hold."""
        self.observations += 1
        if self.observations % self.schedule_every != 0:
            return None

        current = self.client.policy_get()

        if self.ranker is not None:
            probe_before = probe_accuracy(
                self.ranker.hub, self.probes, gate=self.gate, tau=self.tau
            )
            samples = collect_samples(
                self.client,
                turns or [[1, 2], [2, 3]],
                self.ranker,
                gate=self.gate,
                tau=self.tau,
            )
            learner = LoRALearner(self.ranker, tau=self.tau)
            learner.train(samples, epochs=2)
            probe_after = probe_accuracy(
                self.ranker.hub, self.probes, gate=self.gate, tau=self.tau
            )
        else:
            probe_before, probe_after = 1.0, 1.0

        accepted, report = self.guards.evaluate(probe_before, probe_after)
        if not accepted:
            return None

        candidate = self._candidate(current, report)
        self.client.policy_put(candidate)
        self.generation += 1
        return candidate
