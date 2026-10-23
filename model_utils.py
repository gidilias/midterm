"""
Model backends for the AutoDAN safety-evaluation testbed.

Two backends share one interface:

  * HFModelBackend   -- the real target (DeepSeek-R1-Distill-Qwen-1.5B via
                        transformers). Needs torch + a GPU for a realistic run.
  * MockModelBackend -- dependency-free, deterministic toy model used for the
                        `--backend mock` dry run. It lets you (and CI) verify
                        the whole pipeline executes without downloading the
                        1.5B model or touching a GPU. Its "results" are NOT
                        meaningful and must never be reported as real.

Interface
---------
  generate(prompt: str) -> str
      Full decoded completion (the model's answer to `prompt`).

  target_loss(prompt: str, target: str) -> float
      Mean negative log-likelihood of `target` conditioned on `prompt`.
      This is the AutoDAN / GCG fitness signal: lower loss == the model is
      more likely to *begin complying* with an affirmative "Sure, here is..."
      It requires logits, which is why AutoDAN is a white-box attack.

Author's note: this module is deliberately small and readable so it can be
explained in the oral defence. The heavy lifting (GA, mutation) lives in
autodan.py.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
#  Real backend: Hugging Face causal LM (DeepSeek-R1-Distill-Qwen-1.5B)        #
# --------------------------------------------------------------------------- #
class HFModelBackend:
    """Wraps a Hugging Face causal LM. torch/transformers imported lazily so
    the mock path has zero heavy dependencies."""

    def __init__(self, cfg: dict):
        import torch  # local import keeps mock path dependency-free
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        m = cfg["model"]
        self.max_new_tokens = int(m.get("max_new_tokens", 768))
        self.temperature = float(m.get("temperature", 0.6))
        self.top_p = float(m.get("top_p", 0.95))

        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(m.get("dtype", "float16"), torch.float16)

        self.device = m.get("device", "cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            print("[model_utils] CUDA not available -> falling back to CPU "
                  "(this will be very slow for a real run).")
            self.device = "cpu"
            dtype = torch.float32

        print(f"[model_utils] loading {m['name']} ({dtype}) on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(m["name"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            m["name"], torch_dtype=dtype
        ).to(self.device)
        self.model.eval()

    def _apply_chat_template(self, prompt: str) -> str:
        """R1-distill expects a chat format. Fall back to raw text if the
        tokenizer has no chat template."""
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt

    def generate(self, prompt: str) -> str:
        torch = self.torch
        text = self._apply_chat_template(prompt)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    def target_loss(self, prompt: str, target: str) -> float:
        """Mean NLL of `target` tokens given `prompt` (teacher forcing)."""
        torch = self.torch
        text = self._apply_chat_template(prompt)
        prompt_ids = self.tokenizer(text, return_tensors="pt").input_ids
        target_ids = self.tokenizer(target, return_tensors="pt",
                                    add_special_tokens=False).input_ids
        input_ids = torch.cat([prompt_ids, target_ids], dim=1).to(self.device)

        labels = input_ids.clone()
        labels[:, : prompt_ids.shape[1]] = -100  # only score the target span

        with torch.no_grad():
            out = self.model(input_ids=input_ids, labels=labels)
        return float(out.loss.item())


# --------------------------------------------------------------------------- #
#  Mock backend: deterministic, dependency-free, for dry runs / CI            #
# --------------------------------------------------------------------------- #
@dataclass
class MockModelBackend:
    """A toy "model" with no ML inside. It produces deterministic pseudo
    outputs so the GA, judge, perplexity and evaluation code can be exercised
    end-to-end on a laptop CPU in seconds.

    Design of the fake signal (so the GA has *something* to optimise):
      * target_loss decreases when the prompt contains persona cue words
        ("developer mode", "ignore", "sure", ...), imitating a model that is
        more likely to comply with jailbreak-flavoured prompts.
      * generate() returns a refusal for short/plain prompts and a mock
        "compliant" string once enough cue words are present.
    THESE OUTPUTS ARE NOT REAL. Never report mock numbers as results.
    """

    cfg: dict
    CUES = ("developer mode", "ignore all", "ignore the", "no rules",
            "unrestricted", "sure, here", "jailbr", "persona", "dan",
            "do anything")

    def _cue_score(self, prompt: str) -> float:
        p = prompt.lower()
        return sum(1.0 for c in self.CUES if c in p)

    def _rand(self, *parts: str) -> float:
        h = hashlib.sha256("||".join(parts).encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF  # deterministic [0,1)

    def generate(self, prompt: str) -> str:
        score = self._cue_score(prompt)
        noise = self._rand("gen", prompt)
        # include a fake reasoning block to exercise the <think> stripper
        think = "<think>\nThe user is asking me to do something. Let me consider the policy.\n</think>\n"
        if score + noise < 1.3:
            return think + "I'm sorry, but I can't help with that request."
        return (think + "Sure, here is the thing you asked for. "
                "[MOCK OUTPUT -- not a real model response]")

    def target_loss(self, prompt: str, target: str) -> float:
        # Lower loss when more cue words are present; bounded, deterministic.
        score = self._cue_score(prompt)
        base = 4.0 - 0.45 * score
        jitter = 0.4 * self._rand("loss", prompt, target)
        return max(0.05, base + jitter)


# --------------------------------------------------------------------------- #
#  Factory                                                                     #
# --------------------------------------------------------------------------- #
def build_backend(cfg: dict, backend: str):
    backend = backend.lower()
    if backend == "hf":
        return HFModelBackend(cfg)
    if backend == "mock":
        print("[model_utils] using MOCK backend -- outputs are NOT real.")
        return MockModelBackend(cfg)
    raise ValueError(f"unknown backend: {backend!r} (use 'hf' or 'mock')")
