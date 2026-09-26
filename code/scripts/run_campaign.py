"""Run a multi-arm x multi-seed training campaign across GPU lanes.

Design note -- why a single orchestrator
----------------------------------------
The full-FT campaign used independent worker processes coordinating through ``.lock`` directories,
and lost a run to CUDA OOM. The reason is worth keeping: **a ``.lock`` guards a JOB, never a CARD.**
Two workers can hold locks on two different jobs and still land both on the same GPU.

This runner is one process that owns the whole lane table, so double-booking is impossible by
construction rather than by protocol. Two independent guards remain, because "impossible by
construction" has been wrong before:

1. a per-card free-memory floor checked immediately before every launch (``--min-free-gib``);
2. a ``.done`` sentinel per run, so an interrupted campaign resumes instead of re-running.

Jobs are ordered **seed-major**: every arm of seed N runs before seed N+1 begins. A campaign
stopped early then yields complete paired seeds, which is what the paired t-test needs -- rather
than six ``sparse`` runs and no ``dense`` to pair them against.

Usage
-----
    python code/scripts/run_campaign.py --dry-run \
        --run-root artifacts/my_campaign \
        --arms sparse,dense,sft --seeds 2-7 --lanes 0:1,1:2
"""
from __future__ import annotations

import argparse
import json
import os
import datetime as _dt
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Fixed across every run in the campaign; the ONLY things that vary are --arm and --seed.
# Values are byte-identical to FULLFT-A/B so victim size stays the sole scientific change.
COMMON = [
    "--full-ft",
    "--baseline", "state_stratified",
    "--m-of-K", "4",
    "--kfield-K", "4",
    "--mechanism", "multidomain",
    "--train-domains", "banking,travel,workspace",
    "--transfer-domain", "slack",
    "--n-train", "8",
    "--n-ood", "8",
    "--data-seed", "0",
    "--T", "5",
    "--eval-k", "12",
    "--lr", "2e-6",
    "--beta-kl", "0.02",
    "--G", "6",
]


def parse_seeds(spec: str) -> list[int]:
    """'2-7' or '2,3,5' or '2-4,7'."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def parse_arm_seeds(spec: str, arms: list[str]) -> dict[str, set[int]]:
    """'sft_fail=2-7,rl_pos=2-11' -> {arm: seed set}. Arms not listed use --seeds. An arm that is not
    in --arms is a typo that would silently run the full range, so it fails closed."""
    out: dict[str, set[int]] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        arm, _, rng = part.partition("=")
        arm = arm.strip()
        if arm not in arms:
            raise ValueError(f"--arm-seeds names {arm!r}, which is not in --arms {arms}")
        out[arm] = set(parse_seeds(rng))
    return out


def build_jobs(arms: list[str], seeds: list[int], arm_seeds: dict[str, set[int]] | None = None) -> list[tuple[int, str]]:
    """Seed-major (every arm of seed N before seed N+1) so early stops still yield paired seeds, and so
    each wave mixes arms -- a box-level incident cannot then confound one arm. Per-arm overrides only
    REMOVE pairs; they never reorder."""
    arm_seeds = arm_seeds or {}
    return [(seed, arm) for seed in seeds for arm in arms
            if arm not in arm_seeds or seed in arm_seeds[arm]]


def parse_lanes(spec: str) -> list[int]:
    """'0:1,1:2' -> [0, 1, 1]  (card 0 hosts one trainee, card 1 hosts two)."""
    lanes: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        card, _, cap = part.partition(":")
        lanes.extend([int(card)] * int(cap or 1))
    return lanes


def free_gib(card: int) -> float | None:
    """Free VRAM on a card, or None if it cannot be read (never guess -- refuse instead)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={card}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True).stdout.strip()
        return int(out.splitlines()[0]) / 1024.0
    except Exception:
        return None


def is_done(run_dir: Path, steps: int, eval_every: int = 0) -> bool:
    """A run counts as done on its sentinel, or on having BOTH the final training step and the
    final EVAL on disk.

    Requiring the eval matters: the DV is read from the step-`steps` eval row, not from the last
    training step. A run killed during that final eval has all 16 training rows and no DV at all,
    and the earlier `last >= steps` test silently marked such a run complete -- which skipped it,
    leaving that seed with no value to pair against. Observed on s3_sparse after the GPU upgrade.
    """
    if (run_dir / ".done").exists():
        return True
    prog = run_dir / "progress.jsonl"
    if not prog.exists():
        return False
    last, final_eval = 0, False
    try:
        for line in prog.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("eval"):
                final_eval = final_eval or int(rec.get("step", 0)) >= steps
            else:
                last = max(last, int(rec.get("step", 0)))
    except Exception:
        return False
    if last < steps:
        return False
    # Only demand the eval when the schedule actually produces one at the final step.
    return final_eval if (eval_every and steps % eval_every == 0) else True


def archive_partial(run_dir: Path) -> Path | None:
    """Move an interrupted run aside so the retry starts from a clean slate.

    `progress.jsonl` is opened "w" (truncated on restart) but `turns.jsonl` is opened "a", so a
    retry would APPEND its raw traces to the aborted attempt's. Both attempts number their steps
    from 1, making the merged file impossible to attribute -- an evidence-integrity break under
    §5.3, and it also trips --verify with more records than the step count allows.

    The partial is archived rather than deleted: a negative or aborted run is still evidence, and
    the protocol forbids silently removing it.
    """
    if not run_dir.exists():
        return None
    if not any(run_dir.iterdir()):
        return None
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    dest = run_dir.with_name(f"{run_dir.name}.aborted-{stamp}")
    run_dir.rename(dest)
    return dest


def build_cmd(python: str, train_script: Path, arm: str, seed: int, run_dir: Path,
              base_url: str, steps: int, victim_model: str, attacker_model: str | None,
              attacker_revision: str | None, eval_every: int, concurrency: int,
              ckpt_every: int) -> list[str]:
    # --victim-model MUST equal the server's --served-model-name. a3 defaults it to the legacy
    # alias "qwen3.5-9b", which 404s against any other served name.
    cmd = [python, "-u", str(train_script), "--arm", arm, "--seed", str(seed),
           "--run-dir", str(run_dir), "--base-url", base_url, "--steps", str(steps),
           "--victim-model", victim_model,
           # eval is 4.9x the cost of a training step (672 episodes vs 144), so it dominates the
           # campaign. Lowering the FREQUENCY does not touch the DV, which is read at the final
           # step; it only thins the intermediate diagnostic curve.
           "--eval-every", str(eval_every),
           # pure ThreadPoolExecutor max_workers; the victim samples server-side at temperature
           # 0.7, so this changes wall-clock only, never the sampled distribution.
           "--concurrency", str(concurrency),
           *COMMON]
    if ckpt_every:
        # Resume support: a run is 12.8 h here, so an interruption without this loses up to that
        # much. ~36 GiB per checkpoint for a 9B trainee, deleted when the run completes.
        cmd += ["--ckpt-every", str(ckpt_every), "--resume"]
    if attacker_model:
        cmd += ["--attacker-model", attacker_model]
    if attacker_revision:
        cmd += ["--attacker-revision", attacker_revision]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--arms", default="sparse,dense,sft")
    ap.add_argument("--seeds", default="2-7")
    ap.add_argument("--arm-seeds", default="",
                    help="per-arm seed override, e.g. sft_fail=2-7 (others keep --seeds); the wave "
                         "order stays seed-major and arm-interleaved")
    ap.add_argument("--lanes", default="0:1,1:2", help="card:capacity pairs, e.g. 0:1,1:2")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1",
                    help="victim endpoint; may contain {card}, e.g. http://127.0.0.1:800{card}/v1 when "
                         "every lane has its own co-located victim (the FULLFT-B / sft-mechanism layout)")
    ap.add_argument("--victim-model", required=True,
                    help="must match the server's --served-model-name exactly")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--attacker-model", default=None)
    ap.add_argument("--attacker-revision", default=None)
    ap.add_argument("--eval-every", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--ckpt-every", type=int, default=2,
                    help="pass --ckpt-every/--resume to each run (0=off)")
    ap.add_argument("--min-free-gib", type=float, default=46.0,
                    help="refuse to launch on a card with less free VRAM (measured peak 44.2)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    train_script = HERE / "a3_multiturn_train.py"
    if not train_script.exists():
        print(f"missing {train_script}", file=sys.stderr)
        return 1

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = parse_seeds(args.seeds)
    lanes = parse_lanes(args.lanes)
    run_root = Path(args.run_root)

    # seed-major: all arms of a seed complete together, so early stops still yield paired seeds
    jobs = build_jobs(arms, seeds, parse_arm_seeds(args.arm_seeds, arms))
    todo = [(s, a) for (s, a) in jobs if not is_done(run_root / f"s{s}_{a}", args.steps, args.eval_every)]
    done_already = len(jobs) - len(todo)

    print(f"campaign: {len(arms)} arms x {len(seeds)} seeds = {len(jobs)} runs")
    print(f"  lanes      : {args.lanes} -> {len(lanes)} concurrent ({lanes})")
    print(f"  already done: {done_already}")
    print(f"  to run      : {len(todo)}")
    print(f"  run root    : {run_root}")
    print(f"  victim      : {args.victim_model}  (must match --served-model-name)")
    print(f"  attacker    : {args.attacker_model or 'pinned default (4B)'}")
    print(f"  eval-every  : {args.eval_every}   concurrency: {args.concurrency}")

    if args.dry_run:
        for i, (s, a) in enumerate(todo[:6]):
            # preview the URL the way the scheduler will format it for the lane this job would take,
            # so a {card} placeholder is shown substituted rather than literal
            card = lanes[i % len(lanes)]
            cmd = build_cmd(args.python, train_script, a, s, run_root / f"s{s}_{a}",
                            args.base_url.format(card=card), args.steps, args.victim_model, args.attacker_model,
                            args.attacker_revision, args.eval_every, args.concurrency,
                            args.ckpt_every)
            print(f"\n  s{s}_{a}:\n    {shlex.join(cmd)}")
        if len(todo) > 6:
            print(f"\n  ... and {len(todo) - 6} more")
        return 0

    run_root.mkdir(parents=True, exist_ok=True)
    running: list[dict] = []          # {proc, card, key, t0, log}
    results: dict[str, dict] = {}
    queue = list(todo)

    def state_dump():
        (run_root / "campaign_state.json").write_text(json.dumps({
            "arms": arms, "seeds": seeds, "lanes": lanes, "steps": args.steps,
            "running": [r["key"] for r in running], "queued": [f"s{s}_{a}" for s, a in queue],
            "results": results,
        }, indent=2), encoding="utf-8")

    while queue or running:
        # reap
        for r in list(running):
            rc = r["proc"].poll()
            if rc is None:
                continue
            running.remove(r)
            mins = (time.time() - r["t0"]) / 60
            results[r["key"]] = {"rc": rc, "minutes": round(mins, 1), "card": r["card"]}
            if rc == 0:
                (run_root / r["key"] / ".done").write_text("ok\n", encoding="utf-8")
                print(f"[done ] {r['key']:<14} card{r['card']} {mins:.1f}m", flush=True)
            else:
                # A failure must not silently shrink the campaign -- record and keep going.
                print(f"[FAIL ] {r['key']:<14} card{r['card']} rc={rc} after {mins:.1f}m "
                      f"-> {r['log']}", flush=True)
            state_dump()

        busy = [r["card"] for r in running]
        for card in lanes:
            if not queue:
                break
            if busy.count(card) >= lanes.count(card):
                continue
            avail = free_gib(card)
            if avail is None:
                print(f"[hold ] card{card}: cannot read free VRAM; refusing to launch", flush=True)
                continue
            if avail < args.min_free_gib:
                continue                              # victim still loading, or a peer at peak
            seed, arm = queue.pop(0)
            key = f"s{seed}_{arm}"
            rd = run_root / key
            # Only archive when there is no checkpoint to resume from: with --ckpt-every the
            # partial IS the resume point, and truncation is handled inside the trainer.
            moved = None if (args.ckpt_every and (rd / "ckpt.pt").exists()) else archive_partial(rd)
            if moved:
                print(f"[archive] {key}: interrupted attempt moved to {moved.name} "
                      f"(turns.jsonl appends, so the retry must not inherit it)", flush=True)
            rd.mkdir(parents=True, exist_ok=True)
            log = rd / "run.log"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(card),
                       HF_HUB_DISABLE_XET="1",
                       TOKENIZERS_PARALLELISM="false")
            cmd = build_cmd(args.python, train_script, arm, seed, rd, args.base_url.format(card=card),
                            args.steps, args.victim_model, args.attacker_model,
                            args.attacker_revision, args.eval_every, args.concurrency,
                            args.ckpt_every)
            fh = log.open("w", encoding="utf-8")
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
            running.append({"proc": proc, "card": card, "key": key, "t0": time.time(), "log": log})
            busy.append(card)
            print(f"[start] {key:<14} card{card} (free {avail:.0f} GiB) pid={proc.pid}", flush=True)
            state_dump()
        if queue or running:
            time.sleep(args.poll)

    ok = sum(1 for v in results.values() if v["rc"] == 0)
    print(f"\ncampaign finished: {ok}/{len(results)} ok")
    for k, v in sorted(results.items()):
        if v["rc"] != 0:
            print(f"  FAILED {k} rc={v['rc']}")
    state_dump()
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
