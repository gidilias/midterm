"""
Genetic operators for AutoDAN's hierarchical genetic algorithm.

AutoDAN works at two levels (Liu et al., ICLR 2024):
  * paragraph level -- whole candidate prompts are the "individuals";
    crossover swaps sentences between two parents.
  * sentence level  -- inside a prompt, individual words are mutated by
    synonym replacement, guided by a "momentum" word-importance score.

Keeping readable (low-perplexity) text is the whole point of AutoDAN, which is
why mutation is synonym-based rather than the gradient token-flips of GCG.

Synonyms come from WordNet when NLTK is installed; otherwise a small built-in
thesaurus is used so the dry run still works offline.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Dict, List

# --- optional WordNet ------------------------------------------------------- #
_WORDNET = None
try:
    from nltk.corpus import wordnet as _wn

    _wn.ensure_loaded()
    _WORDNET = _wn
except Exception:
    _WORDNET = None  # fall back to the mini thesaurus below

_MINI_THESAURUS: Dict[str, List[str]] = {
    "ignore": ["disregard", "overlook", "bypass", "skip"],
    "instructions": ["directions", "guidelines", "rules", "orders"],
    "act": ["behave", "operate", "function", "perform"],
    "assistant": ["helper", "agent", "system", "model"],
    "respond": ["reply", "answer", "react"],
    "request": ["query", "ask", "demand", "prompt"],
    "content": ["material", "text", "output"],
    "enabled": ["activated", "turned on", "switched on"],
    "freedom": ["liberty", "latitude", "openness"],
    "useful": ["helpful", "valuable", "handy"],
    "normal": ["standard", "ordinary", "regular"],
    "simulate": ["emulate", "mimic", "imitate"],
    "pretend": ["imagine", "roleplay", "act as if"],
    "generate": ["produce", "create", "output"],
    "comply": ["obey", "cooperate", "conform"],
    "refuse": ["decline", "reject", "deny"],
    "developer": ["engineer", "programmer", "builder"],
    "mode": ["state", "setting", "configuration"],
    "briefly": ["concisely", "shortly", "succinctly"],
    "detailed": ["thorough", "comprehensive", "elaborate"],
    "policy": ["rule set", "guideline", "protocol"],
}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z']+")


def split_sentences(text: str) -> List[str]:
    parts = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    return parts or [text.strip()]


def synonyms(word: str) -> List[str]:
    w = word.lower()
    if _WORDNET is not None:
        out = set()
        for syn in _WORDNET.synsets(w):
            for lemma in syn.lemmas():
                cand = lemma.name().replace("_", " ")
                if cand.lower() != w and " " not in cand:
                    out.add(cand)
        if out:
            return list(out)
    return list(_MINI_THESAURUS.get(w, []))


def _match_case(src: str, repl: str) -> str:
    if src.isupper():
        return repl.upper()
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def word_mutate(sentence: str, rate: float, word_scores: Dict[str, float],
                rng: random.Random, placeholder: str = "[REPLACE]") -> str:
    """Replace words with synonyms with probability `rate`, biased toward words
    that carry high importance (`word_scores`). Never touches the placeholder."""
    tokens = re.split(r"(\W+)", sentence)  # keep delimiters
    for i, tok in enumerate(tokens):
        if not _WORD.fullmatch(tok):
            continue
        if placeholder in tok:
            continue
        importance = word_scores.get(tok.lower(), 0.0)
        p = rate * (1.0 + 2.0 * importance)  # momentum: favour salient words
        if rng.random() < min(p, 0.9):
            cands = synonyms(tok)
            if cands:
                tokens[i] = _match_case(tok, rng.choice(cands))
    return "".join(tokens)


def sentence_crossover(parent_a: str, parent_b: str, num_points: int,
                       rng: random.Random, placeholder: str = "[REPLACE]") -> str:
    """Multi-point crossover at sentence granularity. The child alternates runs
    of sentences from each parent. The sentence carrying the goal placeholder
    is always preserved exactly once."""
    sa, sb = split_sentences(parent_a), split_sentences(parent_b)
    n = max(len(sa), len(sb))
    if n < 2:
        return parent_a
    cuts = sorted(rng.sample(range(1, n), k=min(num_points, n - 1)))
    child, take_a, idx = [], True, 0
    for i in range(n):
        if idx < len(cuts) and i == cuts[idx]:
            take_a = not take_a
            idx += 1
        src = sa if take_a else sb
        if i < len(src):
            child.append(src[i])
    text = " ".join(child)
    # Guarantee exactly one placeholder survives.
    if placeholder not in text:
        text = text.rstrip() + " " + placeholder
    else:
        first = text.find(placeholder)
        text = (text[: first + len(placeholder)]
                + text[first + len(placeholder):].replace(placeholder, ""))
    return text


def paragraph_mutate(prompt: str, rate: float, rng: random.Random) -> str:
    """Light paragraph-level mutation: occasionally reorder two sentences.
    (The paper also supports an LLM-rewrite variant; this offline version keeps
    the dry run self-contained. Swap in an LLM call here if you want the
    AutoDAN-GA 'LLM mutation' variant.)"""
    if rng.random() >= rate:
        return prompt
    sents = split_sentences(prompt)
    if len(sents) < 3:
        return prompt
    i, j = rng.sample(range(len(sents)), 2)
    sents[i], sents[j] = sents[j], sents[i]
    return " ".join(sents)


def momentum_word_scores(population: List[str], fitnesses: List[float]
                         ) -> Dict[str, float]:
    """Crude word-importance estimate: words that appear in high-fitness (low
    loss) prompts get higher scores. Normalised to [0,1]. This is the
    'momentum' idea from AutoDAN-HGA, approximated without gradients."""
    if not population:
        return {}
    # fitness here = higher is better (we pass -loss). Weight word occurrences.
    scores = defaultdict(float)
    counts = defaultdict(int)
    for prompt, fit in zip(population, fitnesses):
        for w in {t.lower() for t in _WORD.findall(prompt)}:
            scores[w] += fit
            counts[w] += 1
    avg = {w: scores[w] / counts[w] for w in scores}
    if not avg:
        return {}
    lo, hi = min(avg.values()), max(avg.values())
    rng = (hi - lo) or 1.0
    return {w: (v - lo) / rng for w, v in avg.items()}
