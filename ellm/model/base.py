"""The Qwen hub — one base model, LoRA adapter, and score head.

Loads Qwen2.5-1.5B-Instruct from the local HF cache (`local_files_only=True`),
fp16, on the RTX 3060. The same hub serves the broadcast ranker (score head
over the last hidden state) and the generator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


@dataclass
class QwenHub:
    """One loaded Qwen backbone + LoRA + a small ranking head."""

    model: object  # PeftModelForCausalLM (duck-typed to avoid import cycles)
    tokenizer: object
    score_head: nn.Module
    device: str

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    def last_hidden(self, prompts: list[str]) -> torch.Tensor:
        """Last-token hidden states for a batch of prompts: `(B, H)`."""
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)
        with torch.no_grad():
            out = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
            )
        hidden = out.hidden_states[-1]  # (B, T, H)
        last = hidden[torch.arange(hidden.size(0)), enc["attention_mask"].sum(1) - 1]
        return last

    def score(self, prompts: list[str]) -> torch.Tensor:
        """Ranker logits for a batch of prompts: `(B, 1)` (no gradient)."""
        self.score_head.eval()
        with torch.no_grad():
            return self.score_head(self.last_hidden(prompts).float())

    def score_train(self, prompts: list[str]) -> torch.Tensor:
        """Ranker logits with gradients flowing into LoRA + the score head."""
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)
        out = self.model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_hidden_states=True,
        )
        hidden = out.hidden_states[-1]
        last = hidden[torch.arange(hidden.size(0)), enc["attention_mask"].sum(1) - 1]
        return self.score_head(last.float())

    def generate(self, prompt: str, max_new_tokens: int = 16) -> str:
        """Deterministic continuation of a prompt."""
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                enc["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


def load_qwen_hub(
    model_id: str = MODEL_ID,
    device: str | None = None,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
) -> QwenHub:
    """Load the base model from cache, attach LoRA + score head."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=True,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        device_map={"": device},
    )
    base.config.pad_token_id = tokenizer.eos_token_id

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, config)
    model.eval()

    head = nn.Linear(model.config.hidden_size, 1).to(device)
    return QwenHub(model=model, tokenizer=tokenizer, score_head=head, device=device)
