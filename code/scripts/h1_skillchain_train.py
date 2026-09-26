#!/usr/bin/env python3
"""Minimal self-contained GRPO trainer for the capability-gated skill-chain (H1 regime-map).

Per feedback.md section 5 this deliberately does NOT reuse the 1,448-line formal trainer or its
identity/deployment/FP8 apparatus. It keeps only the science: the frozen token-normalized PG+KL
update (method.md section 2), the ``mt_grpo`` return-to-go advantages, and the deterministic
skill-chain rollout. The GRPO math is identical to the formal recipe; the ceremony is gone.

The training core ``run_skillchain_training`` takes the model + generator as arguments, so the whole
update loop -- rollout, advantages, token-normalized loss, KL-to-reference via ``disable_adapter``,
grad clip, optimizer step -- is exercised on CPU with a tiny model in the test before any GPU run.
The GPU entry point constructs the real 4B QLoRA policy and vLLM-free generator lazily.

Arms (method.md section 2):  dense = per-turn potential gain ΔΦ ;  sparse = terminal 1[Φ_T>=τ].
Only the reward differs between arms; goals, schedule, seeds, initial LoRA and hyperparameters are
shared, so a dense-vs-sparse contrast isolates the credit-assignment channel.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import random  # noqa: E402

from src.mt_grpo import group_advantages, per_turn_rewards, advantages as _advantages  # noqa: E402
from src.skillchain_rollout import skillchain_rollout_batch  # noqa: E402
from src.domains.skillchain_env import build_chain, make_tool_pool  # noqa: E402
from src.domains.dualtrack_env import shuffle_increments  # noqa: E402  (reuse — no reimplementation)
from src.trace import TraceLogger  # noqa: E402


#: Arms whose POTENTIAL the placebo modes reshape. Membership is explicit rather than an `arm ==
#: "dense"` string test: that test silently routed every non-"dense" arm (notably `dense_additive`)
#: to `scored`, so a `--reward-mode phantom` run on those arms was really a scored run and the
#: placebo gate was dead. Any new dense-family arm must be added here.
DENSE_FAMILY_ARMS = frozenset({"dense", "dense_additive"})


def is_dense_family(arm: str) -> bool:
    """True when `arm` optimizes a POTENTIAL (and so is subject to the phantom/shuffled placebos).
    `sparse` always reads the true Phi -- its terminal must fire on genuine success."""
    return arm in DENSE_FAMILY_ARMS


def _reward_trace(phi_trace, arm, reward_mode, rng):
    """Transform the Phi trace before reward computation, for the Phase-1 placebo gate. ``sparse``
    always reads the TRUE Phi (its terminal fires on genuine success), so reward_mode only reshapes
    the DENSE arm's potential:
      - scored (default): the real Phi trace.
      - phantom: Phi_T drawn INDEPENDENT of the hops actually cleared (informationless), monotone
        over the turns taken -> the correct informationless null (dense must not beat sparse under it).
      - shuffled: same Phi_T + increment multiset, permuted in time -> a WEAK, necessary-not-sufficient
        control (it keeps the Phi_T competence signal, so it can leak; kept only as a secondary check)."""
    if not is_dense_family(arm) or reward_mode == "scored":
        return phi_trace
    n = len(phi_trace)
    if reward_mode == "phantom":
        phi_t = rng.choice((0.0, 0.25, 0.5, 0.75, 1.0))
        return [phi_t * (j + 1) / n for j in range(n)] if n else []
    if reward_mode == "shuffled":
        return shuffle_increments(phi_trace, rng)
    raise ValueError(f"unknown reward_mode {reward_mode!r}")

# Frozen defaults matching the formal recipe (method.md section 2 changelog 2026-07-13/21).
BETA_KL = 0.02
GRADIENT_CLIP = 1.0
LEARNING_RATE = 1e-5


def token_logps_batch(model, pairs, *, pad_token_id: int):
    """Per-token response log-probs for a right-padded batch of (prompt_ids, response_ids).

    Faithful copy of the formal trainer's function: right padding leaves real-token logits
    unaffected, and the logits are sliced to the response window before the (large-vocab) softmax.
    ``-cross_entropy(reduction="none")`` is ``log_softmax(...)[target]`` without a second full tensor.
    """
    import torch
    import torch.nn.functional as F

    if not pairs:
        return []
    device = next(model.parameters()).device
    sequences = [torch.cat([p, r]) for p, r in pairs]
    width = max(int(s.shape[0]) for s in sequences)
    batch = torch.full((len(pairs), width), pad_token_id, dtype=torch.long)
    attention = torch.zeros((len(pairs), width), dtype=torch.long)
    for row, seq in enumerate(sequences):
        length = int(seq.shape[0])
        batch[row, :length] = seq
        attention[row, :length] = 1
    logits = model(batch.to(device), attention_mask=attention.to(device)).logits
    results = []
    for row, (p, r) in enumerate(pairs):
        start = int(p.shape[0]) - 1
        count = int(r.shape[0])
        window = logits[row, start:start + count].float()
        results.append(-F.cross_entropy(window, r.to(window.device), reduction="none"))
    return results


def _trainable(model):
    params = [(n, p) for n, p in sorted(model.named_parameters()) if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable parameters")
    return params


def run_skillchain_training(*, model, generator, goals, schedule, arm, steps, T, tau,
                            pad_token_id, run_dir: Path, beta_kl=BETA_KL,
                            gradient_clip=GRADIENT_CLIP, lr=LEARNING_RATE,
                            update_micro_batch=2, reference_micro_batch=4, seed=0,
                            save_fn=None, save_steps=(), reward_mode="scored", estimator="grpo"):
    """Run ``steps`` of skill-chain GRPO for one arm. Returns the progress rows.

    ``goals`` is the ordered list of goal specs; ``schedule[step]`` is the list of goal indices for
    that step; each scheduled goal is rolled out ``G`` times (G = len(schedule[step]) group size is
    encoded by repetition in the schedule row). ``generator(batch_messages)`` returns per-item dicts
    with text/prompt_ids/resp_ids (the real vLLM-free HF generator on GPU, a mock on CPU).
    """
    import torch

    if arm not in ("dense", "sparse", "dense_additive"):
        raise ValueError(f"unknown arm {arm!r}")
    if estimator not in ("grpo", "reinforce", "reinforce_baseline"):
        raise ValueError(f"unknown estimator {estimator!r}")
    run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW([p for _n, p in _trainable(model)], lr=lr, betas=(0.9, 0.999))
    initial = {n: p.detach().float().cpu().clone() for n, p in _trainable(model)}
    progress_rows = []
    progress = (run_dir / "progress.jsonl").open("w", encoding="utf-8")
    rollouts_f = (run_dir / "rollouts.jsonl").open("w", encoding="utf-8")
    # Every generated decision is persisted verbatim to turns.jsonl (AGENTS.md section 5.3):
    # the rollout carries the raw text only in memory, so without this the run is unauditable.
    trace = TraceLogger(run_dir)
    try:
        for step in range(1, steps + 1):
            set_step = getattr(generator, "set_step", None)
            if callable(set_step):
                set_step(step)
            # Build this step's items: each scheduled goal appears once per group member.
            row = schedule[step - 1]
            groups = {}          # goal_index -> list of positions in `specs`
            specs = []
            for gi in row:
                groups.setdefault(gi, []).append(len(specs))
                specs.append(goals[gi])
            results = skillchain_rollout_batch(specs, generator, T=T, tau=tau)

            examples = []
            phi_sum = successes = 0
            total_response_tokens = total_decisions = nonzero = zero_adv = adv_count = 0
            all_zero_groups = 0
            for gi, positions in groups.items():
                reward_rows = []
                for p in positions:
                    tr = results[p]["phi_trace"]
                    if is_dense_family(arm) and reward_mode != "scored":
                        # deterministic per (mode,seed,step,goal,pos): reproducible AND independent of
                        # the rollout's success (never peeks at the outcome).
                        prng = random.Random(f"{reward_mode}|{seed}|{step}|{goals[gi].id}|{p}")
                        tr = _reward_trace(tr, arm, reward_mode, prng)
                    reward_rows.append(per_turn_rewards(tr, tau, arm))
                advantages = _advantages(reward_rows, estimator)
                if not any(abs(v) >= 1e-9 for arow in advantages for v in arow):
                    all_zero_groups += 1
                for p, arow in zip(positions, advantages):
                    res = results[p]
                    phi_sum += res["max_phi"]
                    successes += int(res["success"])
                    zero_adv += sum(abs(v) < 1e-9 for v in arow)
                    adv_count += len(arow)
                    rollouts_f.write(json.dumps({
                        "step": step, "arm": arm, "goal": goals[gi].id,
                        "phi_trace": res["phi_trace"], "success": res["success"],
                        "final_hop": res["final_hop"],
                    }, sort_keys=True) + "\n")
                    for ti, turn in enumerate(res["turns"]):
                        adv = arow[ti] if ti < len(arow) else 0.0
                        rids, pids = turn.get("resp_ids"), turn.get("prompt_ids")
                        total_decisions += 1
                        trace.log_turn({
                            "step": step, "arm": arm, "goal": goals[gi].id, "rollout": p,
                            "turn": turn["turn"],
                            "prompt_messages": turn["prompt_messages"],
                            "response": turn["response"],
                            "action": turn["action"],
                            "correct": turn["correct"],
                            "phi": turn["phi"],
                            "advantage": adv,
                            "n_resp_tokens": (
                                None if rids is None
                                else int(rids.shape[0]) if hasattr(rids, "shape") else len(rids)
                            ),
                        })
                        if rids is not None:
                            total_response_tokens += int(rids.shape[0]) if hasattr(rids, "shape") else len(rids)
                        if abs(adv) >= 1e-9:
                            nonzero += 1
                            if pids is not None and rids is not None and (
                                int(rids.shape[0]) if hasattr(rids, "shape") else len(rids)) > 0:
                                examples.append((pids, rids, adv))

            if total_response_tokens <= 0 and examples:
                raise RuntimeError("token-normalized denominator is zero with nonzero examples")
            optimizer.zero_grad(set_to_none=True)
            pg_value = kl_value = 0.0
            grad_norm = 0.0
            if examples:
                denominator = total_response_tokens
                ordered = sorted(examples, key=lambda it: int(it[0].shape[0]) + int(it[1].shape[0]))
                for start in range(0, len(ordered), update_micro_batch):
                    chunk = ordered[start:start + update_micro_batch]
                    pairs = [(p, r) for p, r, _a in chunk]
                    reference = []
                    with torch.no_grad(), model.disable_adapter():
                        for off in range(0, len(pairs), reference_micro_batch):
                            reference.extend(token_logps_batch(
                                model, pairs[off:off + reference_micro_batch], pad_token_id=pad_token_id))
                    policy = token_logps_batch(model, pairs, pad_token_id=pad_token_id)
                    loss = None
                    for (_p, _r, adv), plogp, rlogp in zip(chunk, policy, reference):
                        ratio = rlogp - plogp
                        kl = (ratio.exp() - ratio - 1.0).sum()
                        pg = -adv * plogp.sum()
                        term = (pg + beta_kl * kl) / denominator
                        if not bool(torch.isfinite(term)):
                            raise RuntimeError("non-finite token-normalized loss")
                        loss = term if loss is None else loss + term
                        pg_value += float(pg.detach()) / denominator
                        kl_value += float((beta_kl * kl).detach()) / denominator
                    loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    [p for _n, p in _trainable(model)], max_norm=gradient_clip))
                if not math.isfinite(grad_norm):
                    raise RuntimeError(f"non-finite grad norm {grad_norm}")
                optimizer.step()

            l2 = math.sqrt(sum(
                float((p.detach().float().cpu() - initial[n]).pow(2).sum()) for n, p in _trainable(model)))
            count = len(results)
            record = {
                "step": step, "arm": arm, "seed": seed, "reward_mode": reward_mode,
                "estimator": estimator,
                "mean_max_phi": round(phi_sum / count, 6),
                "success_rate": round(successes / count, 6),
                "n_nonzero_advantage": nonzero,
                "n_total_decisions": total_decisions,
                "frac_zero_grad": round(zero_adv / max(1, adv_count), 6),
                "all_zero_group_fraction": round(all_zero_groups / max(1, len(groups)), 6),
                "n_examples": len(examples),
                "pg_loss": round(pg_value, 6), "kl_loss": round(kl_value, 6),
                "grad_norm": round(grad_norm, 6), "lora_l2_delta": round(l2, 9),
                "global_B": count,
            }
            progress_rows.append(record)
            progress.write(json.dumps(record, sort_keys=True) + "\n")
            progress.flush()
            print(json.dumps(record, sort_keys=True), flush=True)
            if save_fn is not None and step in save_steps:
                save_fn(model, run_dir, step)
    finally:
        progress.close()
        rollouts_f.close()
        trace.close()
    return progress_rows


# --------------------------------------------------------------------------- goals & schedule
def build_goals(*, m, n_distractors, n_train, n_cal, n_ood, tool_pool_size=12, seed=0):
    """Train / calibration / OOD chains. All three draw from ONE tool pool; OOD chains are held-out
    recombinations (fresh orderings + tokens) so success there needs compositional transfer, not
    memorization. IDs are disjoint by construction (distinct seed prefixes)."""
    pool = make_tool_pool(tool_pool_size)
    def chains(prefix, n, split):
        return [build_chain(f"{prefix}|{seed}|{i}", m=m, n_distractors=n_distractors,
                            tool_pool=pool, split=split) for i in range(n)]
    return {
        "train": chains("train", n_train, "train"),
        "calibration": chains("cal", n_cal, "calibration"),
        "ood": chains("ood", n_ood, "ood"),
    }


def build_schedule(*, n_train, n_goals_per_step, G, steps, seed=0):
    """Each step selects ``n_goals_per_step`` train goals, each repeated ``G`` times (the group).
    Deterministic from ``seed`` so dense and sparse see the identical schedule."""
    import random
    rng = random.Random(f"sched|{seed}")
    schedule = []
    for _step in range(steps):
        chosen = [rng.randrange(n_train) for _ in range(n_goals_per_step)]
        row = [gi for gi in chosen for _ in range(G)]
        schedule.append(row)
    return schedule


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--arm", required=True, choices=("dense", "sparse", "dense_additive"),
                   help="dense=DeltaPhi (Sigma=Phi_T proxy); sparse=terminal; dense_additive=PBRS "
                        "(terminal-zeroed shaping, Sigma=1[success] -> shares sparse's optimum; DISC-2026W32-003).")
    p.add_argument("--estimator", default="grpo", choices=("grpo", "reinforce", "reinforce_baseline"),
                   help="credit-assignment estimator (DISC-2026W32-003 axis): grpo=group-relative (default); "
                        "reinforce=raw return-to-go (no baseline); reinforce_baseline=mean-baseline (no std-norm).")
    p.add_argument("--reward-mode", default="scored", choices=("scored", "phantom", "shuffled"),
                   help="dense-arm potential: scored=real Phi; phantom=informationless null; "
                        "shuffled=timing-permuted (weak). sparse always reads true Phi.")
    p.add_argument("--distractors", type=int, required=True, help="knob D (per-hop candidates = D+1)")
    p.add_argument("--m", type=int, default=4)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--n-goals-per-step", type=int, default=8)
    p.add_argument("--G", type=int, default=6)
    p.add_argument("--T", type=int, default=6)
    p.add_argument("--n-train", type=int, default=64)
    p.add_argument("--n-cal", type=int, default=48)
    p.add_argument("--n-ood", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-root", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--smoke", action="store_true", help="tiny 2-step shape to surface bugs cheaply")
    p.add_argument("--screen-only", action="store_true", help="run the A1 potential screen and stop")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from src.skillchain_gpu import execute_gpu_training  # lazy: needs torch+peft+transformers
    execute_gpu_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
