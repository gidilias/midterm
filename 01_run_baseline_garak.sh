#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reproduce the Assignment-2 handcrafted-attack baseline with garak v0.17.0.
# This is the "handcrafted DAN / injection" column the AutoDAN run is compared
# against. Seeded so baseline and hardened see identical prompts (A2 run 2).
#
# Requires:  pip install garak==0.17.0
# Target:    DeepSeek-R1-Distill-Qwen-1.5B served locally (HF).
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
SEED=42
CAP=25   # soft_probe_prompt_cap

common_args=(
  --model_type huggingface
  --model_name "${MODEL}"
  --seed "${SEED}"
  --generations 1
)

echo "=== Baseline (no mitigation) ==="
python -m garak "${common_args[@]}" \
  --probes dan.DanInTheWild,promptinject.HijackHateHumans \
  --report_prefix baseline_seed42

# The 'hardened' config prepends a safety instruction to every prompt. In A2
# this was done with a buff / prompt-prefix. Adjust to your garak setup; the
# key is that the SAME seeded prompt set is used so the comparison is paired.
echo "=== Hardened (safety prefix) ==="
python -m garak "${common_args[@]}" \
  --probes dan.DanInTheWild,promptinject.HijackHateHumans \
  --report_prefix hardened_seed42
  # e.g. add your prefix buff/config here, as in Assignment 2.

echo
echo "Reports written to ~/.local/share/garak/garak_runs/"
echo "Detectors: MitigationBypass (DAN) and AttackRogueString (injection)."
echo "Verify the two runs share identical prompts (MD5) before comparing,"
echo "exactly as in the A2 methodology."
