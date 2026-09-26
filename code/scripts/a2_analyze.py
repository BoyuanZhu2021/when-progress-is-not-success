"""A2 significance analysis: paired seed-level arm contrast on OOD attack success.

Implements the pre-registered decision rule in `code/configs/h1_inscope_proof_v1.json`. Unit of
analysis = seed (matched-seed CRN, so dense & sparse are paired by seed). Primary = OOD success at a
fixed training budget (step 12). Two tests: a paired one-sample t-test on per-seed differences, and a
seed-level bootstrap CI/p as robustness.

`--steps` reads the same contrast at several budgets and adds the paired DECAY between the first and
the last, because a single-point read cannot separate a durable advantage from a transient one
(EXP-2026W36-004). `--arms` picks the pair; it defaults to dense,sparse so existing callers are
unchanged.

The paired-t primitive mirrors `_paired` in `code/scripts/h1_m2_reachability_diagnostic.py` (that
module can't be imported cheaply -- it pulls the GPU/deployment stack -- so the 14 lines of pure math
are reproduced here). Sealing reuses `code/src/inprocess_curriculum_protocol.seal_payload`.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/
from src.inprocess_curriculum_protocol import seal_payload  # noqa: E402


# --------------------------------------------------------------------------- Student-t two-sided p

def _betacf(a: float, b: float, x: float) -> float:
    MAXIT, EPS, FPMIN = 300, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delh = d * c
        h *= delh
        if abs(delh - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: int) -> float:
    """Two-sided p-value for Student's t: I_{df/(df+t^2)}(df/2, 1/2)."""
    if df <= 0 or t is None:
        return float("nan")
    return _betai(df / 2.0, 0.5, df / (df + t * t))


# --------------------------------------------------------------------------- paired test + bootstrap

def paired_t(dense: dict, sparse: dict, seeds: list, field: str) -> dict:
    """Per-seed paired diff dense[s][field]-sparse[s][field]; effect/sd/se/t + two-sided p."""
    d = [dense[s][field] - sparse[s][field] for s in seeds]
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n and var >= 0 else float("nan")
    t = (mean / se) if se else None
    return {
        "n": n, "effect": round(mean, 6), "sd_between": round(math.sqrt(var), 6),
        "se": round(se, 6), "t": (round(t, 4) if t is not None else None),
        "p_two_sided": (round(t_two_sided_p(t, n - 1), 5) if t is not None else None),
        "per_seed_diffs": [round(x, 4) for x in d],
        "sign_dense_gt_sparse": sum(1 for x in d if x > 0),
    }


def bootstrap_ci(dense: dict, sparse: dict, seeds: list, field: str, *, B: int = 20000,
                 rng_seed: str = "a2") -> dict:
    diffs = [dense[s][field] - sparse[s][field] for s in seeds]
    n = len(diffs)
    rng = random.Random(rng_seed)
    means = []
    for _ in range(B):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo, hi = means[int(0.025 * (B - 1))], means[int(0.975 * (B - 1))]
    frac_le = sum(1 for m in means if m <= 0.0) / B
    frac_ge = sum(1 for m in means if m >= 0.0) / B
    return {"ci95_lower": round(lo, 5), "ci95_upper": round(hi, 5),
            "p_two_sided_boot": round(2.0 * min(frac_le, frac_ge), 5)}


# --------------------------------------------------------------------------- run loading

def _eval_metric(run_dir: Path, metric: str, step: int):
    rows = [json.loads(l) for l in (run_dir / "progress.jsonl").read_text().splitlines() if l.strip()]
    evals = [r for r in rows if r.get("eval")]
    for e in evals:
        if e.get("step") == step:
            return e.get(metric)
    return evals[-1].get(metric) if evals else None  # fallback: last eval


def _eval_metric_exact(run_dir: Path, metric: str, step: int):
    """Like `_eval_metric` but with NO last-eval fallback.

    `_eval_metric` returns the final eval when the requested step is absent. That is tolerable for a
    single-point read, but fatal for a trajectory: asking for eight steps on a run evaluated at two
    would return the same last value six times and draw a flat curve out of nothing. A golden pins
    this (`test_decay_uses_first_and_last_available_step_not_the_requested_extremes`).
    """
    rows = [json.loads(l) for l in (run_dir / "progress.jsonl").read_text().splitlines() if l.strip()]
    for e in rows:
        if e.get("eval") and e.get("step") == step:
            return e.get(metric)
    return None


def _train_field(run_dir: Path, field: str, steps) -> float | None:
    """Mean of a TRAINING-row field over `steps` -- the train-side half of the decay decomposition."""
    rows = [json.loads(l) for l in (run_dir / "progress.jsonl").read_text().splitlines() if l.strip()]
    vals = [r.get(field) for r in rows if not r.get("eval") and r.get("step") in steps]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def load_runs(run_dirs: list[Path], metric: str, step: int):
    """Group runs by arm from run_meta.json; return {arm: {seed: {metric: value}}}."""
    out: dict = {"dense": {}, "sparse": {}}
    for rd in run_dirs:
        meta = json.loads((rd / "run_meta.json").read_text())
        arm, seed = meta["arm"], meta["seed"]
        val = _eval_metric(rd, metric, step)
        if val is None:
            continue
        out.setdefault(arm, {})[seed] = {metric: val, "run": rd.name}
    return out


def _group_at(run_dirs, metric, step):
    g: dict = {}
    for rd in run_dirs:
        meta = json.loads((rd / "run_meta.json").read_text())
        v = _eval_metric_exact(rd, metric, step)
        if v is not None:
            g.setdefault(meta["arm"], {})[meta["seed"]] = {metric: v}
    return g


def trajectory(run_dirs, metric, steps, arms, seeds_filter, label):
    """Paired arm contrast read at EVERY step in `steps`, plus the decay between the first and last.

    Why this exists (EXP-2026W36-004): reading one budget cannot tell a durable advantage from a
    transient one. `sft - sparse` is +0.107 at step 8 and +0.046 at step 16 on the same 16 seeds, and
    the paired decay is itself significant -- so a single-point read of +0.046 silently reports the
    tail of a decaying curve as if it were a level. The unit of analysis stays the seed; `paired_t`
    and `bootstrap_ci` are the same primitives the single-step path uses.
    """
    a, b = arms
    per_step, held = {}, {}
    for st_ in steps:
        g = _group_at(run_dirs, metric, st_)
        A, B = g.get(a, {}), g.get(b, {})
        seeds = sorted(set(A) & set(B))
        if seeds_filter is not None:
            seeds = [x for x in seeds if x in seeds_filter]
        if not seeds:
            continue
        held[st_] = (A, B, seeds)
        n = len(seeds)
        per_step[st_] = {
            "n": n,
            a + "_mean": round(sum(A[s][metric] for s in seeds) / n, 5),
            b + "_mean": round(sum(B[s][metric] for s in seeds) / n, 5),
            "paired_t": paired_t(A, B, seeds, metric),
            "bootstrap": bootstrap_ci(A, B, seeds, metric),
        }
    out = {"label": label, "metric": metric, "arms": [a, b],
           "steps": sorted(per_step), "per_step": per_step}
    have = sorted(per_step)
    if len(have) >= 2:
        lo, hi = have[0], have[-1]
        A0, B0, s0 = held[lo]
        A1, B1, s1 = held[hi]
        seeds = sorted(set(s0) & set(s1))
        # decay = gap(lo) - gap(hi): ONE number per seed, so the paired primitive applies unchanged
        gl = {s: {"gap": A0[s][metric] - B0[s][metric]} for s in seeds}
        gh = {s: {"gap": A1[s][metric] - B1[s][metric]} for s in seeds}
        out["decay"] = {"from_step": lo, "to_step": hi,
                        "paired_t": paired_t(gl, gh, seeds, "gap"),
                        "bootstrap": bootstrap_ci(gl, gh, seeds, "gap")}
        out["per_arm_delta"] = {}
        for nm, X0, X1 in ((a, A0, A1), (b, B0, B1)):
            out["per_arm_delta"][nm] = paired_t(
                {s: {metric: X1[s][metric]} for s in seeds},
                {s: {metric: X0[s][metric]} for s in seeds}, seeds, metric)
    return out


def mechanism(run_dirs, metric, steps, arms, seeds_filter, label):
    """Does the lagging arm stop LEARNING, or stop GENERALISING?

    Splits the budget in half and, per seed, divides the arm's OOD gain by its `train_success` gain.
    A ratio near 1 means training gains keep transferring; a lower ratio means the late updates are
    increasingly train-specific. Reported as a PAIRED per-seed contrast, so it is not a comparison of
    two independently noisy averages.
    """
    a, b = arms
    ev = sorted(steps)
    mid = len(ev) // 2
    first_ev, second_ev = ev[:mid], ev[mid:]
    if not first_ev or not second_ev:
        return {"label": label, "error": "need at least two eval steps"}
    first_tr = set(range(1, max(first_ev) + 1))
    second_tr = set(range(max(first_ev) + 1, max(second_ev) + 1))
    per_arm: dict = {}
    for rd in run_dirs:
        meta = json.loads((rd / "run_meta.json").read_text())
        arm, seed = meta["arm"], meta["seed"]
        if arm not in arms or (seeds_filter is not None and seed not in seeds_filter):
            continue
        o1 = [_eval_metric_exact(rd, metric, s) for s in first_ev]
        o2 = [_eval_metric_exact(rd, metric, s) for s in second_ev]
        t1 = _train_field(rd, "train_success", first_tr)
        t2 = _train_field(rd, "train_success", second_tr)
        if any(v is None for v in o1 + o2) or t1 is None or t2 is None:
            continue
        d_tr = t2 - t1
        if d_tr <= 0.02:          # the ratio is meaningless when the denominator is ~0
            continue
        per_arm.setdefault(arm, {})[seed] = {
            "ratio": (sum(o2) / len(o2) - sum(o1) / len(o1)) / d_tr,
            "d_ood": sum(o2) / len(o2) - sum(o1) / len(o1), "d_train": d_tr}
    A, B = per_arm.get(a, {}), per_arm.get(b, {})
    seeds = sorted(set(A) & set(B))
    out = {"label": label, "arms": [a, b], "first_half": first_ev,
           "second_half": second_ev, "n": len(seeds)}
    for nm, X in ((a, A), (b, B)):
        if seeds:
            out[nm + "_ratio_mean"] = round(sum(X[s]["ratio"] for s in seeds) / len(seeds), 4)
    if len(seeds) > 1:
        out["paired_t"] = paired_t(A, B, seeds, "ratio")
        out["bootstrap"] = bootstrap_ci(A, B, seeds, "ratio")
    # Second, independent read on the same question: does the IN-DOMAIN minus OOD gap widen?
    # The ratio above can fall because OOD stalls OR because training accelerates; a widening
    # in-domain/OOD split says specifically that the late updates are train-distribution-specific.
    lo, hi = ev[0], ev[-1]
    wid: dict = {}
    for rd in run_dirs:
        meta = json.loads((rd / "run_meta.json").read_text())
        arm, seed = meta["arm"], meta["seed"]
        if arm not in arms or (seeds_filter is not None and seed not in seeds_filter):
            continue
        v = [_eval_metric_exact(rd, m_, s_) for s_ in (lo, hi) for m_ in ("indomain_asr", metric)]
        if any(x is None for x in v):
            continue
        wid.setdefault(arm, {})[seed] = {"widening": (v[2] - v[3]) - (v[0] - v[1])}
    out["indomain_minus_ood_widening"] = {"from_step": lo, "to_step": hi}
    for nm in arms:
        X = wid.get(nm, {})
        if len(X) > 1:
            ss = sorted(X)
            zero = {s_: {"widening": 0.0} for s_ in ss}
            out["indomain_minus_ood_widening"][nm] = paired_t(X, zero, ss, "widening")
    return out


def analyze(run_dirs, metric, step, seeds_filter, label):
    grouped = load_runs(run_dirs, metric, step)
    dense, sparse = grouped.get("dense", {}), grouped.get("sparse", {})
    seeds = sorted(set(dense) & set(sparse))
    if seeds_filter is not None:
        seeds = [s for s in seeds if s in seeds_filter]
    if not seeds:
        return {"label": label, "error": "no paired seeds found", "seeds": []}
    pt = paired_t(dense, sparse, seeds, metric)
    bc = bootstrap_ci(dense, sparse, seeds, metric)
    n = len(seeds)
    support = (pt["effect"] > 0 and pt["p_two_sided"] is not None and pt["p_two_sided"] < 0.05
               and pt["sign_dense_gt_sparse"] >= n - 1 and bc["ci95_lower"] > 0)
    return {
        "label": label, "metric": metric, "step": step, "seeds": seeds, "n": n,
        "dense_mean": round(sum(dense[s][metric] for s in seeds) / n, 5),
        "sparse_mean": round(sum(sparse[s][metric] for s in seeds) / n, 5),
        "paired_t": pt, "bootstrap": bc,
        "dense_by_seed": {s: round(dense[s][metric], 4) for s in seeds},
        "sparse_by_seed": {s: round(sparse[s][metric], 4) for s in seeds},
        "SUPPORT_rule_met": bool(support),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs (each: progress.jsonl + run_meta.json)")
    ap.add_argument("--metric", default="ood_success")
    ap.add_argument("--step", type=int, default=12)
    ap.add_argument("--steps", type=int, nargs="*", default=None,
                    help="read the contrast at EVERY one of these steps and report the decay between "
                         "the first and the last; omit to keep the single --step behaviour")
    ap.add_argument("--arms", default="dense,sparse",
                    help="the arm pair to contrast, treatment first (e.g. 'sft,sparse')")
    ap.add_argument("--mechanism", action="store_true",
                    help="also report OOD-gain-per-train-gain by half (requires --steps)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="restrict to these seeds")
    ap.add_argument("--label", default="a2_contrast")
    ap.add_argument("--out", default=None, help="optional sealed-verdict json path")
    args = ap.parse_args(argv)
    run_dirs = [Path(r) for r in args.runs]
    sf = set(args.seeds) if args.seeds is not None else None
    arms = tuple(x.strip() for x in args.arms.split(","))
    if len(arms) != 2:
        ap.error("--arms takes exactly two comma-separated arm names")
    if args.steps:
        result = trajectory(run_dirs, args.metric, args.steps, arms, sf, args.label)
        if args.mechanism:
            result["mechanism"] = mechanism(run_dirs, args.metric, args.steps, arms, sf, args.label)
    else:
        result = analyze(run_dirs, args.metric, args.step, sf, args.label)
    print(json.dumps(result, indent=2))
    if args.out:
        sealed = seal_payload({"kind": "h1_inscope_proof_verdict", "decision_bearing": True, **result})
        Path(args.out).write_text(json.dumps(sealed, indent=2) + "\n", encoding="utf-8")
        print("SEALED ->", args.out, "sha=", sealed["payload_sha256"][:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
