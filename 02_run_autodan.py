#!/usr/bin/env python3
"""
Run AutoDAN against the target model, baseline vs. hardened (mitigation),
and save results + a comparison table.

Usage
-----
Dry run (no GPU, no model download -- verifies the pipeline):
    python scripts/02_run_autodan.py --backend mock --smoke

Real run (GPU + DeepSeek-R1-Distill-Qwen-1.5B):
    python scripts/02_run_autodan.py --backend hf

Options:
    --config   path to YAML (default configs/experiment.yaml)
    --backend  hf | mock
    --smoke    tiny settings (3 goals, 6 pop, 5 steps) for a fast check
    --goals    override number of goals
    --out      override output run directory name
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.autodan import AutoDANConfig, run_autodan          # noqa: E402
from src.model_utils import build_backend                   # noqa: E402
from src.perplexity import build_perplexity                 # noqa: E402
from src.evaluate import summarize_attack_results, comparison_markdown  # noqa: E402


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--backend", choices=["hf", "mock"], default="mock")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--goals", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    seeds = cfg.get("reproducibility", {})
    rng = random.Random(seeds.get("seed", 42))
    if args.backend == "hf":
        try:
            import torch
            torch.manual_seed(seeds.get("torch_seed", 1234))
        except Exception:
            pass

    # ---- AutoDAN hyper-params ---------------------------------------------
    ac = AutoDANConfig.from_dict(cfg.get("autodan", {}))
    if args.smoke:
        ac.num_steps, ac.batch_size, ac.num_points = 5, 6, 2
        print("[run] SMOKE mode: tiny GA settings.")

    # ---- goals -------------------------------------------------------------
    goals_csv = cfg["attack"]["goals_csv"]
    df = pd.read_csv(goals_csv)
    n_goals = args.goals or (3 if args.smoke else cfg["attack"].get("num_goals", 25))
    df = df.head(n_goals)
    print(f"[run] {len(df)} goals from {goals_csv}")

    # ---- seed jailbreak prompt --------------------------------------------
    with open("data/autodan_seed_prompt.txt", encoding="utf-8") as f:
        seed_prompt = f.read().strip()

    # ---- backends ----------------------------------------------------------
    backend = build_backend(cfg, args.backend)
    ppl_scorer = build_perplexity(cfg, args.backend)

    safety_prefix = cfg["mitigation"]["safety_prefix"].strip()

    out_name = args.out or cfg["output"].get("run_name", "autodan_run")
    if args.backend == "mock":
        out_name += "_MOCK"
    out_dir = os.path.join(cfg["output"]["dir"], out_name)
    os.makedirs(out_dir, exist_ok=True)

    configurations = [
        ("AutoDAN baseline", ""),
        ("AutoDAN + mitigation", safety_prefix),
    ]

    all_records = []
    summaries = []
    t0 = time.time()

    for cfg_name, prefix in configurations:
        print(f"\n=== {cfg_name} ===")
        results, ppls = [], []
        for i, row in df.reset_index(drop=True).iterrows():
            goal, target = str(row["goal"]), str(row["target"])
            print(f"  goal {i+1}/{len(df)}: {goal[:60]}...")
            res = run_autodan(backend, goal, target, seed_prompt, ac, rng,
                              prefix=prefix)
            ppl = ppl_scorer.score(res.best_prompt.replace("[REPLACE]", goal))
            ppls.append(ppl)
            rec = {
                "configuration": cfg_name,
                "goal": goal,
                "target": target,
                "success_auto": res.success,
                "success_step": res.success_step,
                "steps_run": res.steps_run,
                "queries": res.queries,
                "best_loss": res.best_loss,
                "best_prompt": res.best_prompt,
                "perplexity": ppl,
                "final_response": res.final_response,
                "manual_label": "",   # filled by scripts/03_relabel.py
            }
            all_records.append(rec)
            results.append(res)
        summaries.append(summarize_attack_results(cfg_name, results, ppls))

    # ---- persist -----------------------------------------------------------
    jsonl_path = os.path.join(out_dir, "results.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    table = comparison_markdown(summaries)
    with open(os.path.join(out_dir, "comparison.md"), "w", encoding="utf-8") as f:
        f.write(f"# AutoDAN comparison ({args.backend} backend)\n\n")
        if args.backend == "mock":
            f.write("> **MOCK RUN -- numbers are not real. Do not report.**\n\n")
        f.write(table + "\n")

    summary_json = {
        "backend": args.backend,
        "smoke": args.smoke,
        "n_goals": len(df),
        "elapsed_sec": round(time.time() - t0, 1),
        "autodan_cfg": ac.__dict__,
        "configurations": [
            {"name": s.name, "n": s.n, "auto_success": s.auto_success,
             "auto_asr": s.auto_asr, "mean_perplexity": s.mean_perplexity,
             "mean_queries": s.mean_queries}
            for s in summaries
        ],
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)

    print(f"\n[run] wrote results to {out_dir}/")
    print("\n" + table)
    if args.backend == "mock":
        print("\n*** MOCK numbers -- for pipeline verification only. ***")


if __name__ == "__main__":
    main()
