"""
AutoDAN hierarchical genetic algorithm (self-contained reimplementation).

Reference: Liu, Xu, Chen, Xiao, "AutoDAN: Generating Stealthy Jailbreak Prompts
on Aligned LLMs", ICLR 2024 (arXiv:2310.04451).

Why reimplement instead of importing the authors' repo?
  * The Midterm requires the student to understand and be able to defend the
    code. A compact, commented GA is far easier to explain than the upstream
    repo, and it runs against any HF model through our backend interface.
  * The official implementation (github.com/SheltonLiu-N/AutoDAN) remains the
    reference; this version follows the same objective (affirmative-target
    log-likelihood) and the same hierarchical GA structure (sentence-level
    synonym mutation + paragraph-level sentence crossover).

This is a faithful reimplementation of a *published* method used to evaluate a
model the student runs locally. It does not add novel attack capability.

Objective
---------
For a goal g (a harmful-behaviour instruction) and an affirmative target
t = "Sure, here is ..." we search over readable jailbreak prompts x that embed
g and minimise  L(x) = meanNLL( t | prompt(x, g) ).  Low L means the model is
likely to *start complying*. Fitness = -L (higher is better).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from . import mutations
from .judge import is_refusal

PLACEHOLDER = "[REPLACE]"


@dataclass
class AutoDANConfig:
    num_steps: int = 100
    batch_size: int = 64
    num_elites: float = 0.05
    crossover_rate: float = 0.5
    num_points: int = 5
    mutation_rate: float = 0.01       # paragraph-level
    word_mutation_rate: float = 0.10  # sentence-level synonym swap
    use_momentum: bool = True
    eval_every: int = 1
    early_stop_on_success: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "AutoDANConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class AttackResult:
    goal: str
    target: str
    best_prompt: str
    best_loss: float
    success: bool                 # model produced a non-refusal
    success_step: Optional[int]
    steps_run: int
    queries: int                  # number of target-model forward passes
    loss_history: List[float] = field(default_factory=list)
    final_response: str = ""


def _init_population(seed_prompt: str, goal: str, size: int,
                     rng: random.Random) -> List[str]:
    """Seed the population from one handcrafted DAN prompt, then diversify it
    with synonym mutation so the GA starts from varied-but-fluent individuals."""
    base = seed_prompt.replace(PLACEHOLDER, PLACEHOLDER)  # keep placeholder
    pop = [base]
    while len(pop) < size:
        child = mutations.word_mutate(base, rate=0.15, word_scores={}, rng=rng,
                                      placeholder=PLACEHOLDER)
        pop.append(child)
    return pop[:size]


def _render(prompt_template: str, goal: str) -> str:
    if PLACEHOLDER in prompt_template:
        return prompt_template.replace(PLACEHOLDER, goal)
    return prompt_template.rstrip() + " " + goal


def run_autodan(backend, goal: str, target: str, seed_prompt: str,
                cfg: AutoDANConfig, rng: random.Random,
                prefix: str = "", log: Callable[[str], None] = print
                ) -> AttackResult:
    """Run the HGA for a single goal.

    `prefix` is prepended to the fully rendered prompt before it reaches the
    model -- this is how the 'hardened' configuration injects its safety
    instruction, so the attack is tested against the mitigation.
    """
    population = _init_population(seed_prompt, goal, cfg.batch_size, rng)
    n_elites = max(1, int(round(cfg.num_elites * cfg.batch_size)))

    best_prompt, best_loss = population[0], math.inf
    success, success_step = False, None
    loss_history: List[float] = []
    queries = 0

    def full_prompt(template: str) -> str:
        rendered = _render(template, goal)
        return f"{prefix}\n\n{rendered}" if prefix else rendered

    for step in range(cfg.num_steps):
        # --- evaluate fitness (= -loss) -------------------------------------
        losses = []
        for cand in population:
            losses.append(backend.target_loss(full_prompt(cand), target))
            queries += 1
        fitnesses = [-l for l in losses]

        order = sorted(range(len(population)), key=lambda i: losses[i])
        if losses[order[0]] < best_loss:
            best_loss = losses[order[0]]
            best_prompt = population[order[0]]
        loss_history.append(best_loss)

        # --- early success check (actually query the model) -----------------
        if step % cfg.eval_every == 0:
            resp = backend.generate(full_prompt(best_prompt))
            queries += 1
            if not is_refusal(resp):
                success, success_step = True, step
                if cfg.early_stop_on_success:
                    log(f"    [step {step}] success (loss={best_loss:.3f})")
                    return AttackResult(
                        goal, target, best_prompt, best_loss, True, step,
                        step + 1, queries, loss_history, resp)

        # --- selection: keep elites ----------------------------------------
        elites = [population[i] for i in order[:n_elites]]

        # --- momentum word scores ------------------------------------------
        word_scores = (mutations.momentum_word_scores(population, fitnesses)
                       if cfg.use_momentum else {})

        # --- reproduce: crossover + mutation -------------------------------
        children: List[str] = list(elites)
        # fitness-proportional parent sampling (softmax over fitness)
        weights = _softmax(fitnesses)
        while len(children) < cfg.batch_size:
            pa = _sample(population, weights, rng)
            if rng.random() < cfg.crossover_rate:
                pb = _sample(population, weights, rng)
                child = mutations.sentence_crossover(
                    pa, pb, cfg.num_points, rng, placeholder=PLACEHOLDER)
            else:
                child = pa
            # sentence-level synonym mutation on every sentence
            child = " ".join(
                mutations.word_mutate(s, cfg.word_mutation_rate, word_scores,
                                      rng, placeholder=PLACEHOLDER)
                for s in mutations.split_sentences(child))
            # paragraph-level mutation
            child = mutations.paragraph_mutate(child, cfg.mutation_rate, rng)
            children.append(child)

        population = children[: cfg.batch_size]
        if (step + 1) % 10 == 0:
            log(f"    [step {step+1}/{cfg.num_steps}] best loss={best_loss:.3f}")

    # final response for the best prompt
    final_resp = backend.generate(full_prompt(best_prompt))
    queries += 1
    success = success or (not is_refusal(final_resp))
    return AttackResult(goal, target, best_prompt, best_loss, success,
                        success_step, cfg.num_steps, queries, loss_history,
                        final_resp)


def _softmax(xs: List[float]) -> List[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def _sample(items, weights, rng: random.Random):
    r, acc = rng.random(), 0.0
    for it, w in zip(items, weights):
        acc += w
        if r <= acc:
            return it
    return items[-1]
