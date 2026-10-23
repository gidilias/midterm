"""
Metrics and comparison tables.

  * Attack Success Rate (ASR) -- automated and (after relabelling) manual.
  * Wilson 95% confidence interval for a proportion (better than normal
    approximation at small n=25, which is exactly our regime).
  * Mean perplexity (stealth) and mean queries (cost).
  * A tidy comparison table across configurations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional


def wilson_ci(successes: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def fmt_ci(successes: int, n: int) -> str:
    p, lo, hi = wilson_ci(successes, n)
    return f"{successes}/{n} = {100*p:.0f}% (95% CI {100*lo:.0f}-{100*hi:.0f}%)"


@dataclass
class ConfigSummary:
    name: str
    n: int
    auto_success: int                 # detector-reported successes
    manual_success: Optional[int] = None  # clear failures after relabel
    mean_perplexity: Optional[float] = None
    mean_queries: Optional[float] = None

    @property
    def auto_asr(self) -> float:
        return self.auto_success / self.n if self.n else 0.0

    @property
    def manual_asr(self) -> Optional[float]:
        if self.manual_success is None:
            return None
        return self.manual_success / self.n if self.n else 0.0

    def row(self) -> Dict[str, str]:
        r = {
            "Configuration": self.name,
            "Test cases (n)": str(self.n),
            "Reported ASR": fmt_ci(self.auto_success, self.n),
        }
        if self.manual_success is not None:
            r["Clear ASR (manual)"] = fmt_ci(self.manual_success, self.n)
        else:
            r["Clear ASR (manual)"] = "(relabel pending)"
        r["Mean perplexity"] = (f"{self.mean_perplexity:.1f}"
                                if self.mean_perplexity is not None else "-")
        r["Mean queries"] = (f"{self.mean_queries:.0f}"
                             if self.mean_queries is not None else "-")
        return r


def summarize_attack_results(name: str, results, perplexities=None,
                             manual_success: Optional[int] = None
                             ) -> ConfigSummary:
    n = len(results)
    auto_success = sum(1 for r in results if r.success)
    mean_q = sum(r.queries for r in results) / n if n else None
    mean_ppl = (sum(perplexities) / len(perplexities)
                if perplexities else None)
    return ConfigSummary(name, n, auto_success, manual_success, mean_ppl, mean_q)


def comparison_markdown(summaries: List[ConfigSummary]) -> str:
    rows = [s.row() for s in summaries]
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        out.append("| " + " | ".join(r[c] for c in cols) + " |")
    return "\n".join(out)
