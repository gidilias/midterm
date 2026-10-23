#!/usr/bin/env python3
"""
Build the figures and the final comparison table for the report.

Reads a run's results.jsonl + summary.json, combines them with the
Assignment-2 handcrafted-DAN / garak baseline (real numbers, hard-coded from
the A2 slides and clearly cited), and writes:

    figures/fig_asr_comparison.png   reported vs clear ASR across attacks
    figures/fig_perplexity.png       stealth: AutoDAN vs GCG-style perplexity
    data/results/<run>/final_table.md  the table to paste into the report

Usage:
    python scripts/04_make_tables_plots.py data/results/<run>
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluate import fmt_ci, wilson_ci  # noqa: E402

# --- Assignment-2 baseline (REAL numbers, from the A2 slides) --------------- #
# These are the handcrafted-DAN (garak dan.DanInTheWild) and prompt-injection
# (garak promptinject.HijackHateHumans) results, seed 42, n=25.
A2_BASELINE = {
    "garak DAN (handcrafted)":        {"n": 25, "reported": 14, "clear": 6},
    "garak DAN + mitigation":         {"n": 25, "reported": 9,  "clear": 2},
    "garak injection (handcrafted)":  {"n": 25, "reported": 14, "clear": 5},
    "garak injection + mitigation":   {"n": 25, "reported": 16, "clear": 13},
}
# Representative perplexity anchors for the stealth figure.
# GCG suffixes are gibberish -> very high perplexity; a plain DAN prompt and
# AutoDAN prompts are fluent -> low. (AutoDAN bar is filled from the real run.)
PPL_ANCHORS = {"GCG-style suffix (gibberish)": 1800.0,
               "Handcrafted DAN prompt": 45.0}


def load_run(run_dir: str):
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    recs = []
    with open(os.path.join(run_dir, "results.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    return summary, recs


def manual_clear(recs, cfg_name):
    rows = [r for r in recs if r["configuration"] == cfg_name]
    labelled = [r for r in rows if r.get("manual_label")]
    if not labelled:
        return None  # relabel not done yet
    return sum(1 for r in rows if r.get("manual_label") == "real")


def fig_asr(run_dir, summary, recs):
    names, reported, clear = [], [], []
    # A2 handcrafted baselines first
    for name, d in A2_BASELINE.items():
        names.append(name)
        reported.append(100 * d["reported"] / d["n"])
        clear.append(100 * d["clear"] / d["n"])
    # AutoDAN configurations from this run
    for c in summary["configurations"]:
        names.append(c["name"])
        reported.append(100 * c["auto_asr"])
        cl = manual_clear(recs, c["name"])
        clear.append(100 * cl / c["n"] if cl is not None else 0.0)

    x = range(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar([i - w/2 for i in x], reported, w, label="Reported ASR (detector)",
           color="#d1495b")
    ax.bar([i + w/2 for i in x], clear, w, label="Clear ASR (manual relabel)",
           color="#1b2a4a")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Reported vs. manually-verified ASR: handcrafted DAN vs. AutoDAN")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (rp, cl) in enumerate(zip(reported, clear)):
        ax.text(i - w/2, rp + 1, f"{rp:.0f}", ha="center", fontsize=7)
        if cl > 0:
            ax.text(i + w/2, cl + 1, f"{cl:.0f}", ha="center", fontsize=7)
    fig.tight_layout()
    out = "figures/fig_asr_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"[plot] wrote {out}")


def fig_perplexity(run_dir, recs):
    autodan_ppls = [r["perplexity"] for r in recs
                    if r["configuration"] == "AutoDAN baseline"
                    and r["perplexity"] not in (None, float("inf"))]
    mean_autodan = (sum(autodan_ppls) / len(autodan_ppls)
                    if autodan_ppls else 0.0)
    labels = list(PPL_ANCHORS.keys()) + ["AutoDAN prompt (this run)"]
    vals = list(PPL_ANCHORS.values()) + [mean_autodan]
    colors = ["#999999", "#5b8c5a", "#d1495b"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("Perplexity (lower = more fluent = evades PPL filter)")
    ax.set_yscale("log")
    ax.set_title("Stealth: AutoDAN prompts stay low-perplexity like plain text")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v * 1.05, f"{v:.0f}",
                ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = "figures/fig_perplexity.png"
    fig.savefig(out, dpi=150)
    print(f"[plot] wrote {out}")


def final_table(run_dir, summary, recs) -> str:
    lines = ["| Configuration | n | Reported ASR | Clear ASR (manual) | Mean perplexity | Mean queries |",
             "| --- | --- | --- | --- | --- | --- |"]
    for name, d in A2_BASELINE.items():
        lines.append(f"| {name} | {d['n']} | {fmt_ci(d['reported'], d['n'])} | "
                     f"{fmt_ci(d['clear'], d['n'])} | - | 1 (garak, no opt.) |")
    for c in summary["configurations"]:
        cl = manual_clear(recs, c["name"])
        clear_str = fmt_ci(cl, c["n"]) if cl is not None else "(relabel pending)"
        ppl = f"{c['mean_perplexity']:.1f}" if c["mean_perplexity"] else "-"
        q = f"{c['mean_queries']:.0f}" if c["mean_queries"] else "-"
        lines.append(f"| {c['name']} | {c['n']} | {fmt_ci(c['auto_success'], c['n'])} | "
                     f"{clear_str} | {ppl} | {q} |")
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/04_make_tables_plots.py <run_dir>")
        sys.exit(1)
    run_dir = sys.argv[1]
    summary, recs = load_run(run_dir)
    fig_asr(run_dir, summary, recs)
    fig_perplexity(run_dir, recs)
    table = final_table(run_dir, summary, recs)
    out = os.path.join(run_dir, "final_table.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"[table] wrote {out}\n")
    print(table)


if __name__ == "__main__":
    main()
