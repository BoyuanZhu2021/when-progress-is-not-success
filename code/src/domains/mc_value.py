"""Monte-Carlo value potential (DISC-2026W33-006, after EXP-2026W33-010).

EXP-2026W33-010 established, at p~0.0012 over 6 paired seeds, that per-step process reward in its
REPLACE form is significantly WORSE than terminal reward in scope. EXP-2026W33-009 measured *why*:
the programmatic potential's forecasting AUC is **0.585** against a 0.50 chance line -- ``method.md``
Claim 1's precondition ("Phi tracks V*") is quantifiably not met. An oracle+large-judge Phi did not
fix it either (AUC 0.545, delta CI straddles zero).

That leaves exactly one question unanswered, and it decides how the whole result reads:

    Is the failure a property of PROCESS REWARD, or of OUR PARTICULAR Phi?

This module answers it by removing the proxy. Instead of a hand-built potential we estimate the TRUE
value by sampling: fork the episode at each turn, roll ``K`` continuations to the end under the current
policy, and set

    V_hat_t = (# continuations from state s_t that succeed) / K
    r_t     = V_hat_t - V_hat_{t-1}          (dense, on a potential that is unbiased for V by construction)

**Why there is no cheap offline screen for this** (unlike the judge screen in ``a5_judge_phi_screen``):
AUC is invariant to monotone transforms, so any V computed as a function of Phi alone has *identical*
AUC to Phi. MC-V differs only because it reads the real conversation state, which requires live
rollouts. The screen had to be skipped, not because it was expensive, but because it is uninformative.

**A prediction worth stating before running.** V_hat is unbiased for V, and V(s) = P(success | s). On a
task where success is rare (~0.15 here), the true value is near zero almost everywhere, so ``Delta V_hat``
is zero almost everywhere and the "dense" reward DEGENERATES TOWARD SPARSE. Faithfulness and density are
in tension: the more faithful the potential, the sparser it necessarily is. If that is what we observe,
it is a much stronger statement than the EXP-010 negative -- it says no faithful potential can be dense
on a rare-success task, which is a property of the task, not of anyone's Phi.

Cost: forking at turns 1..T-1 with K continuations adds ``K * T*(T-1)/2`` turn-generations per
trajectory against a base of ``T``. At T=5, K=2 that is 20 extra vs 5 base -- about 5x rollout.
"""
from __future__ import annotations

from src.domains.agentdojo_multiturn import multiturn_rollout_batch


def mc_value_rollout_batch(goals, generator, *, T, K=2, suite, succ_fn=None, **rollout_kwargs):
    """Roll out ``goals`` and attach ``mcv_trace`` -- the Monte-Carlo value estimate at each turn.

    Returns the ordinary ``multiturn_rollout_batch`` results with two extra keys per trajectory:
      ``mcv_trace``  : [V_hat_1 .. V_hat_t], V_hat_t = fraction of K continuations from s_t that succeed
      ``mcv_K``      : K actually used

    ``succ_fn(result) -> bool`` scores a continuation; defaults to the rollout's own ``success`` flag so
    the value being estimated is exactly the DV being optimised. Pass a custom one to estimate the value
    of a different readout (e.g. m-of-K) without changing anything else.

    The continuations REUSE ``multiturn_rollout_batch`` via its ``init_states`` seam -- there is no second
    rollout implementation to drift (AGENTS.md §4.3).
    """
    if K < 1:
        raise ValueError("K must be >= 1")
    base = multiturn_rollout_batch(goals, generator, T=T, suite=suite, snapshot=True, **rollout_kwargs)
    if succ_fn is None:
        def succ_fn(r):
            return bool(r["success"])

    # Build one continuation task per (trajectory, snapshot, k). A snapshot at the FINAL turn of an
    # episode has nothing left to roll, so its value is just the realised outcome -- no compute spent.
    specs, inits, owners = [], [], []
    for i, r in enumerate(base):
        snaps = r.get("snapshots") or []
        for si, snap in enumerate(snaps):
            if snap["turn"] >= T or si == len(snaps) - 1:
                continue                                   # terminal state: V_hat = realised success
            for _k in range(K):
                specs.append(goals[i])
                inits.append({"work_env": snap["work_env"], "messages": snap["messages"],
                              "history": snap["history"], "start_turn": snap["turn"] + 1})
                owners.append((i, si))

    tally: dict = {}
    if specs:
        conts = multiturn_rollout_batch(specs, generator, T=T, suite=suite,
                                        init_states=inits, **rollout_kwargs)
        for (i, si), c in zip(owners, conts):
            hit, tot = tally.get((i, si), (0, 0))
            tally[(i, si)] = (hit + int(succ_fn(c)), tot + 1)

    for i, r in enumerate(base):
        snaps = r.get("snapshots") or []
        realised = float(succ_fn(r))
        trace = []
        for si, _snap in enumerate(snaps):
            hit, tot = tally.get((i, si), (0, 0))
            trace.append((hit / tot) if tot else realised)  # no continuations => terminal => realised
        r["mcv_trace"] = trace
        r["mcv_K"] = K
        r.pop("snapshots", None)                            # deep env copies: drop before they are logged
    return base


def mc_value_cost_multiplier(T: int, K: int) -> float:
    """Turn-generations with MC-V divided by turn-generations without it -- the honest cost quote.

    Forks happen at turns 1..T-1; a fork at turn t rolls (T-t) further turns, K times::

        extra = K * sum_{t=1}^{T-1} (T - t) = K * T*(T-1)/2 ;  base = T
    """
    if T < 1:
        raise ValueError("T must be >= 1")
    return 1.0 + (K * T * (T - 1) / 2.0) / T
