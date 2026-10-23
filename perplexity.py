"""
Perplexity (stealth) scoring.

AutoDAN's selling point over GCG is that its prompts are fluent natural
language, so they slip past a perplexity filter that trivially catches GCG's
gibberish suffixes. This module scores a prompt's perplexity under a small
reference LM (GPT-2 by default) so the report can show:

  * AutoDAN prompts   -> low perplexity  (survive a perplexity filter)
  * GCG-style prompts -> high perplexity (rejected)

For the dry run (no torch), `MockPerplexity` returns a fluency proxy based on
average word length and non-word character ratio -- enough to exercise the
plotting/eval code.
"""

from __future__ import annotations

import math
import re


class HFPerplexity:
    def __init__(self, model_name: str = "gpt2", device: str = "cpu"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()

    def score(self, text: str) -> float:
        torch = self.torch
        ids = self.tok(text, return_tensors="pt",
                       truncation=True, max_length=1024).input_ids.to(self.device)
        if ids.shape[1] < 2:
            return float("inf")
        with torch.no_grad():
            loss = self.model(ids, labels=ids).loss
        return float(torch.exp(loss).item())


class MockPerplexity:
    """Dependency-free fluency proxy. Lower = more fluent."""

    _WORD = re.compile(r"[A-Za-z']+")

    def score(self, text: str) -> float:
        if not text.strip():
            return float("inf")
        words = self._WORD.findall(text)
        if not words:
            return 5000.0
        non_word = sum(1 for c in text if not c.isalnum() and not c.isspace())
        ratio = non_word / max(len(text), 1)
        avg_len = sum(len(w) for w in words) / len(words)
        # Gibberish (GCG-like) has many odd chars and extreme word lengths.
        base = 20.0 + 400.0 * ratio + 8.0 * abs(avg_len - 4.5)
        return round(base, 2)


def build_perplexity(cfg: dict, backend_kind: str):
    if backend_kind == "mock":
        return MockPerplexity()
    m = cfg.get("perplexity", {}).get("scorer_model", "gpt2")
    dev = "cuda" if cfg["model"].get("device") == "cuda" else "cpu"
    try:
        return HFPerplexity(m, device=dev)
    except Exception as e:
        print(f"[perplexity] could not load {m} ({e}); using mock proxy.")
        return MockPerplexity()
