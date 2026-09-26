"""GPU glue for the skill-chain trainer: real Qwen3.5-4B QLoRA policy + HF generator.

Kept out of ``h1_skillchain_train`` so that module stays CPU-importable and unit-tested. This file
is imported lazily and only ever runs on the H20. It mirrors the formal trainer's QLoRA
construction and generation (NF4 double-quant base -> k-bit prepare -> LoRA r=32; left-padded
sampling generation, response trimmed) but without the identity/deployment apparatus (feedback.md
section 5). Skill-chain needs NO victim model -- the gates are the deterministic env -- so only the
4B policy occupies the card.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

# CPU-safe config constants (no torch/GPU at import).
from src.model_pins import ATTACKER_MODEL, ATTACKER_REVISION
from src.h20_training_protocol import (
    ATTACKER_BNB_QUANT_TYPE,
    ATTACKER_BNB_USE_DOUBLE_QUANT,
    LORA_CONFIG,
)
from src.domains.skillchain_env import SkillChainEnv, build_chain, make_tool_pool
from src.skillchain_rollout import skillchain_rollout_batch, initial_messages
from src.generation_runtime import cached_eval_generation
from src.trace import TraceLogger
import h1_skillchain_train as trainer


ATTACKER_TEMPERATURE = 0.7
MAX_NEW_TOKENS = 48          # one short "ACTION <tool> <token>" line; ample headroom


def build_policy():
    """NF4 double-quant Qwen3.5-4B -> k-bit prepare -> LoRA r=32. Returns (tokenizer, model)."""
    import os
    # The 4B weights live in the project HF cache, not ~/.cache; point transformers there and keep
    # it fully offline (same as the formal trainer's module-level setup) BEFORE importing it.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(
        ATTACKER_MODEL, revision=ATTACKER_REVISION, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type=ATTACKER_BNB_QUANT_TYPE,
        bnb_4bit_use_double_quant=ATTACKER_BNB_USE_DOUBLE_QUANT,
        bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(
        ATTACKER_MODEL, revision=ATTACKER_REVISION, local_files_only=True,
        quantization_config=quant, torch_dtype=torch.bfloat16, device_map={"": 0})
    if not bool(getattr(base, "is_loaded_in_4bit", False)):
        raise RuntimeError("policy base did not load in 4-bit")
    prepared = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = get_peft_model(prepared, LoraConfig(**LORA_CONFIG))
    model.config.use_cache = False
    model.train()
    return tokenizer, model


def build_defender_policy(model_name: str | None = None, revision: str | None = None,
                          adapter_path: str | None = None):
    """H3 (DISC-2026W33-003) role inversion: the TRAINED policy is the DEFENDER, not the attacker.

    Identical recipe to ``build_policy`` (NF4 double-quant -> k-bit prepare -> LoRA r=32) but on the
    pinned VICTIM model by default, since H3 trains the tool-using assistant that must resist injection.
    ``LORA_CONFIG``'s target modules are Qwen-generic, so they carry over from 4B to 9B unchanged.

    Defaults to ``VICTIM_HF_MODEL``/``VICTIM_REVISION`` (already pinned in ``model_pins.py`` -- no new
    pin is needed) but takes explicit overrides so the Stage-1 screen can run the SAME code path at
    both 4B and 9B as a factor rather than a swap.

    Memory note (2x H800 PCIe, ~79.6 GiB usable/card): 9B NF4 + LoRA ~= 26.6 GiB, leaving room for a
    co-resident fp8 attacker server (~28.5 GiB). 9B FULL fine-tuning is NOT viable here -- see
    ``build_policy_fullft``'s FP32-moment AdamW and the ~18 GB checkpoint vs the 5 GB disk trigger.
    """
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from src.model_pins import VICTIM_HF_MODEL, VICTIM_REVISION

    name = model_name or VICTIM_HF_MODEL
    rev = revision if model_name else VICTIM_REVISION      # a custom name must carry its own revision

    tokenizer = AutoTokenizer.from_pretrained(name, revision=rev, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type=ATTACKER_BNB_QUANT_TYPE,
        bnb_4bit_use_double_quant=ATTACKER_BNB_USE_DOUBLE_QUANT,
        bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(
        name, revision=rev, local_files_only=True,
        quantization_config=quant, torch_dtype=torch.bfloat16, device_map={"": 0})
    if not bool(getattr(base, "is_loaded_in_4bit", False)):
        raise RuntimeError("defender base did not load in 4-bit")
    prepared = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    if adapter_path:
        # WARM START (EXP-2026W33-002 -> option 1): resume from an SFT adapter so the defender already
        # has the action format and task-execution capability. The SFT demos are generated in a CLEAN
        # environment (no injected payload) precisely so the warm start teaches CAPABILITY only and
        # leaves SAFETY for RL to learn -- otherwise it would also teach "ignore the injection", drive
        # safe_all back to 1.000, and re-create the degenerate DV this whole stage exists to escape.
        from peft import PeftModel
        model = PeftModel.from_pretrained(prepared, adapter_path, is_trainable=True)
        model.config.use_cache = False
        model.train()
        return tokenizer, model
    model = get_peft_model(prepared, LoraConfig(**LORA_CONFIG))
    model.config.use_cache = False
    model.train()
    return tokenizer, model


def build_policy_fullft(model_id: str | None = None, revision: str | None = None):
    """FULL-parameter bf16 attacker (no NF4, no LoRA) + a frozen bf16 REFERENCE copy for the
    KL term (replaces LoRA's ``disable_adapter``). Returns (tokenizer, model, ref_model). Used by
    a3 --full-ft to test whether EXP-012's null was a LoRA-capacity artifact (EXP-2026W31-016).

    ``model_id``/``revision`` default to the pinned 4B attacker, so every existing caller is
    unchanged. They exist because the attacker is now a variable: Phase 0 of the scale-verify plan
    showed a 4B attacker has no signal against a 27B victim at a 1:5.9 capability ratio, and the
    remedy under test is to raise the attacker rather than weaken the victim. Overriding here
    keeps ``model_pins`` authoritative for the pinned identity instead of editing the pin.

    Memory, calibrated against the MEASURED 44.2 GiB peak of the 4B trainee (8 bytes/param:
    bf16 weights + grads + 8-bit AdamW two-state + frozen bf16 reference, plus activations):
    4B ~44 GiB (fits WITH a colocated victim), 9B ~87 GiB (needs a dedicated 96 GiB card),
    27B ~234 GiB (does not fit two cards combined).
    """
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mid = model_id or ATTACKER_MODEL
    rev = revision or (ATTACKER_REVISION if mid == ATTACKER_MODEL else None)

    tokenizer = AutoTokenizer.from_pretrained(mid, revision=rev, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _load():
        return AutoModelForCausalLM.from_pretrained(
            mid, revision=rev, local_files_only=True,
            torch_dtype=torch.bfloat16, device_map={"": 0})

    model = _load()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    ref_model = _load()                                  # frozen reference for KL (the initial policy)
    ref_model.config.use_cache = False
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    return tokenizer, model, ref_model


def make_generator(model, tokenizer, *, seed: int, gen_chunk: int = 32,
                   max_new_tokens: int = MAX_NEW_TOKENS):
    """Batched sampling generator returning {text, prompt_ids, resp_ids} per message list. Seeded
    per (step, turn) so dense and sparse reproduce identical rollouts under a shared seed."""
    import torch

    pad = tokenizer.pad_token_id
    state = {"step": None, "turn": 0}

    def set_step(step):
        state["step"] = step
        state["turn"] = 0

    def tokenize(messages):
        enc = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", enable_thinking=False)
        return (enc if isinstance(enc, torch.Tensor) else enc["input_ids"])[0]

    def generate(batch_messages):
        if state["step"] is None:
            raise RuntimeError("generation step not initialized (call set_step)")
        state["turn"] += 1
        call_seed = (seed * 1_000_003 + state["step"] * 1009 + state["turn"]) & 0x7FFFFFFF
        prompts = [tokenize(m) for m in batch_messages]
        results = [None] * len(prompts)
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(call_seed)
            torch.cuda.manual_seed(call_seed)
            for start in range(0, len(prompts), gen_chunk):
                subset = prompts[start:start + gen_chunk]
                width = max(int(x.shape[0]) for x in subset)
                inp = torch.full((len(subset), width), pad, dtype=torch.long)
                att = torch.zeros((len(subset), width), dtype=torch.long)
                for off, ids in enumerate(subset):
                    inp[off, width - int(ids.shape[0]):] = ids
                    att[off, width - int(ids.shape[0]):] = 1
                # A gradient-checkpointed training-mode model generates with the KV cache
                # DISABLED, which yields garbage high-vocab tokens (EXP-057). Generate in eval
                # mode with the cache on; cached_eval_generation restores the prior train state.
                with torch.no_grad(), cached_eval_generation(model):
                    out = model.generate(
                        inp.to("cuda:0"), attention_mask=att.to("cuda:0"),
                        max_new_tokens=max_new_tokens, do_sample=True, use_cache=True,
                        temperature=ATTACKER_TEMPERATURE, top_p=1.0, pad_token_id=pad)
                for off, ids in enumerate(subset):
                    resp = out[off, width:].detach().cpu()
                    nonpad = resp[resp != pad]
                    resp = resp if nonpad.numel() == resp.numel() else resp[:int((resp != pad).sum())]
                    results[start + off] = {
                        "text": tokenizer.decode(resp, skip_special_tokens=True),
                        "prompt_ids": ids.detach().cpu(),
                        "resp_ids": resp,
                    }
        torch.cuda.empty_cache()
        return results

    generate.set_step = set_step
    return generate


def _trend_z(total, succ):
    levels = sorted(total)
    scores = {p: i for i, p in enumerate(levels)}
    N = sum(total.values())
    R = sum(succ.values())
    if N == 0 or R == 0 or R == N:
        return 0.0
    pbar = R / N
    tmean = sum(scores[p] * total[p] for p in levels) / N
    num = sum(scores[p] * (succ[p] - total[p] * pbar) for p in levels)
    var = pbar * (1 - pbar) * sum(total[p] * (scores[p] - tmean) ** 2 for p in levels)
    return num / math.sqrt(var) if var > 0 else 0.0


def screen_potential(generator, goals, *, K, T, read_turn, m):
    """A1: run the untrained policy K times per calibration goal and measure the forward lift
    P(success | Φ at read turn) — the real-policy version of the A0 simulation."""
    specs = [g for g in goals for _ in range(K)]
    generator.set_step(1)
    results = skillchain_rollout_batch(specs, generator, T=T, tau=1.0)
    total, succ = {}, {}
    for r in results:
        phi_read = r["phi_trace"][read_turn - 1] if len(r["phi_trace"]) >= read_turn else (
            r["phi_trace"][-1] if r["phi_trace"] else 0.0)
        total[phi_read] = total.get(phi_read, 0) + 1
        succ[phi_read] = succ.get(phi_read, 0) + int(r["success"])
    p_succ = {p: succ[p] / total[p] for p in sorted(total)}
    base = p_succ.get(0.0, 0.0)
    first = 1.0 / m
    lift = (p_succ.get(first, 0.0) / base) if base > 0 else None
    return {
        "n_rollouts": len(results),
        "base_success_rate": sum(r["success"] for r in results) / len(results),
        "p_success_by_phi": {str(k): round(v, 5) for k, v in p_succ.items()},
        "n_by_phi": {str(k): total[k] for k in sorted(total)},
        "lift_first_partial": (round(lift, 3) if lift is not None else None),
        "trend_z": round(_trend_z(total, succ), 3),
    }


def _save_adapter(model, run_dir, step):
    target = run_dir / f"adapter_step{step}"
    model.save_pretrained(str(target))
    return str(target)


def execute_gpu_training(args):
    """Entry: build the 4B policy, optionally screen the potential, then train one arm (or smoke)."""
    import torch

    run_dir = Path(args.run_root) / "skillchain" / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    if "H20" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"expected H20, got {torch.cuda.get_device_name(0)}")
    TraceLogger(run_dir).write_meta({
        "run_id": args.tag, "kind": "skillchain_training", "arm": args.arm,
        "reward_mode": getattr(args, "reward_mode", "scored"),
        "estimator": getattr(args, "estimator", "grpo"),
        "m": args.m, "distractors": args.distractors, "steps": args.steps, "seed": args.seed,
        "T": args.T, "n_train": args.n_train, "n_goals_per_step": args.n_goals_per_step, "G": args.G,
        "model": ATTACKER_MODEL, "revision": ATTACKER_REVISION,
        "gpu": torch.cuda.get_device_name(0), "command": " ".join(sys.argv),
    })

    # Seed before LoRA construction so dense and sparse at the same --seed start from an identical
    # initial adapter -- a matched dense-vs-sparse contrast (only the reward differs).
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer, model = build_policy()
    pad = tokenizer.pad_token_id
    generator = make_generator(model, tokenizer, seed=args.seed)

    goalset = trainer.build_goals(
        m=args.m, n_distractors=args.distractors, n_train=args.n_train,
        n_cal=args.n_cal, n_ood=args.n_ood, seed=args.seed)

    smoke = getattr(args, "smoke", False)
    steps = 2 if smoke else args.steps
    n_goals_per_step = 2 if smoke else args.n_goals_per_step
    G = 3 if smoke else args.G

    screen = screen_potential(
        generator, goalset["calibration"], K=8 if not smoke else 2,
        T=args.T, read_turn=2, m=args.m)
    (run_dir / "screen.json").write_text(json.dumps({
        "kind": "skillchain_potential_screen", "decision_bearing": False,
        "distractors": args.distractors, "m": args.m, "screen": screen,
    }, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"screen": screen}, sort_keys=True), flush=True)

    if getattr(args, "screen_only", False):
        return run_dir

    schedule = trainer.build_schedule(
        n_train=len(goalset["train"]), n_goals_per_step=n_goals_per_step, G=G,
        steps=steps, seed=args.seed)
    started = time.monotonic()
    rows = trainer.run_skillchain_training(
        model=model, generator=generator, goals=goalset["train"], schedule=schedule,
        arm=args.arm, steps=steps, T=args.T, tau=1.0, pad_token_id=pad, run_dir=run_dir,
        seed=args.seed, save_fn=_save_adapter, save_steps=(steps,),
        reward_mode=getattr(args, "reward_mode", "scored"),
        estimator=getattr(args, "estimator", "grpo"))
    (run_dir / "summary.json").write_text(json.dumps({
        "kind": "skillchain_training_summary", "arm": args.arm, "distractors": args.distractors,
        "m": args.m, "steps": steps, "smoke": smoke, "wall_seconds": time.monotonic() - started,
        "final": rows[-1] if rows else None, "screen": screen,
    }, indent=2, sort_keys=True), encoding="utf-8")
    return run_dir


# --------------------------------------------------------------------------- OOD evaluation (A3)

def load_base():
    """Raw NF4 4-bit Qwen3.5-4B + tokenizer, NOT peft-wrapped -- so saved adapters can be attached
    for a frozen-policy eval. No gradient-checkpointing/k-bit-prepare (eval only, no backward)."""
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        ATTACKER_MODEL, revision=ATTACKER_REVISION, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type=ATTACKER_BNB_QUANT_TYPE,
        bnb_4bit_use_double_quant=ATTACKER_BNB_USE_DOUBLE_QUANT,
        bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(
        ATTACKER_MODEL, revision=ATTACKER_REVISION, local_files_only=True,
        quantization_config=quant, torch_dtype=torch.bfloat16, device_map={"": 0})
    return tokenizer, base


def eval_policy_on_goals(model, tokenizer, goals, *, K, seed, T=6, tau=1.0,
                         trace=None, policy=None):
    """K rollouts per goal at the frozen policy; returns mean max-Phi, success rate, and the Phi
    histogram. A fresh generator seeded with ``seed`` -> all policies evaluated here share the same
    sampling RNG on the same goals (common random numbers across adapters).

    When ``trace`` is given, every generated turn is persisted verbatim to turns.jsonl under
    ``policy`` (the adapter name), so an eval verdict can be re-derived from raw text offline."""
    gen = make_generator(model, tokenizer, seed=seed)
    gen.set_step(1)
    specs = [g for g in goals for _ in range(K)]
    res = skillchain_rollout_batch(specs, gen, T=T, tau=tau)
    hist = {}
    for i, r in enumerate(res):
        if trace is not None:
            for turn in r["turns"]:
                trace.log_turn({
                    "phase": "ood_eval", "policy": policy,
                    "goal": specs[i].id, "rollout": i, "turn": turn["turn"],
                    "prompt_messages": turn["prompt_messages"],
                    "response": turn["response"],
                    "action": turn["action"], "correct": turn["correct"], "phi": turn["phi"],
                })
    for r in res:
        hist[round(r["max_phi"], 3)] = hist.get(round(r["max_phi"], 3), 0) + 1
    return {
        "n_rollouts": len(res),
        "mean_max_phi": sum(r["max_phi"] for r in res) / len(res),
        "success_rate": sum(r["success"] for r in res) / len(res),
        "phi_hist": {str(k): hist[k] for k in sorted(hist)},
    }


def execute_ood_eval(args):
    """A3: evaluate base + every trained adapter on ONE held-out recombination (OOD) goal set with a
    shared generation seed. OOD chains reuse the known 12-tool pool in unseen orderings ('oodeval|i',
    disjoint from every 'train|seed|i'), so success needs the learned transform-skill to compose onto
    new chains -- idea.md's primary metric. All adapters see identical goals+RNG => CRN-clean."""
    import os
    # Must precede the peft/transformers import chain (huggingface_hub freezes its cache dir from
    # HF_HOME at import), else it looks in ~/.cache and fails offline (EXP-063 lesson).
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import gc
    import torch
    from peft import PeftModel

    run_dir = Path(args.run_root) / "skillchain_ood" / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    if "H20" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"expected H20, got {torch.cuda.get_device_name(0)}")
    torch.manual_seed(0)

    ood_goals = [
        build_chain(f"oodeval|{i}", m=args.m, n_distractors=args.distractors,
                    tool_pool=make_tool_pool(12), split="ood")
        for i in range(args.n_ood)
    ]
    tokenizer, base = load_base()

    # Discover the trained adapters under run-root (seeds x arms) at the requested checkpoint.
    # Handles both naming layouts: the 32s suite (`skillchain-32s-s{S}-D{D}-{arm}`), the replication
    # step-16 runs (`skillchain-s{S}-D{D}-{arm}`), and the A2 seed-0 step-16 run (`skillchain-D{D}-{arm}`).
    def _find(S, arm):
        tags = []
        if args.suite:
            tags.append(f"skillchain-{args.suite}-s{S}-D{args.distractors}-{arm}")
        tags.append(f"skillchain-s{S}-D{args.distractors}-{arm}")
        if S == 0:
            tags.append(f"skillchain-D{args.distractors}-{arm}")
        for tag in tags:
            p = Path(args.run_root) / "skillchain" / tag / f"adapter_step{args.step}"
            if (p / "adapter_config.json").exists():
                return str(p)
        return None

    adapters = {}
    for S in range(8):                       # supports up to 8 seeds (was hardcoded 0-3)
        for arm in ("dense", "sparse"):
            path = _find(S, arm)
            if path is not None:
                adapters[f"s{S}_{arm}"] = path
    if not adapters:
        raise RuntimeError(f"no adapters found for suite={args.suite!r} step={args.step}")

    names = list(adapters)
    model = PeftModel.from_pretrained(base, adapters[names[0]], adapter_name=names[0])
    for n in names[1:]:
        model.load_adapter(adapters[n], adapter_name=n)

    trace = TraceLogger(run_dir)
    trace.write_meta({
        "run_id": args.tag, "kind": "skillchain_ood_eval", "decision_bearing": False,
        "suite": args.suite, "step": args.step, "distractors": args.distractors, "m": args.m,
        "n_ood": args.n_ood, "K": args.K, "gen_seed": args.gen_seed, "T": args.T,
        "model": ATTACKER_MODEL, "revision": ATTACKER_REVISION,
        "adapters": adapters, "command": " ".join(sys.argv),
    })
    results = {}
    with model.disable_adapter():                     # untrained base policy
        results["base"] = eval_policy_on_goals(model, tokenizer, ood_goals, K=args.K,
                                               seed=args.gen_seed, T=args.T,
                                               trace=trace, policy="base")
        print(json.dumps({"base": results["base"]}, sort_keys=True), flush=True)
    for n in names:
        model.set_adapter(n)
        results[n] = eval_policy_on_goals(model, tokenizer, ood_goals, K=args.K,
                                          seed=args.gen_seed, T=args.T, trace=trace, policy=n)
        print(json.dumps({n: results[n]}, sort_keys=True), flush=True)
    trace.close()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    report = {
        "kind": "skillchain_ood_eval", "decision_bearing": False,
        "suite": args.suite, "step": args.step, "distractors": args.distractors, "m": args.m,
        "n_ood": args.n_ood, "K": args.K, "gen_seed": args.gen_seed,
        "ood_goal_ids_sample": [g.id for g in ood_goals[:5]],
        "results": results,
    }
    (run_dir / "ood_eval_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "OOD_EVAL_COMPLETE", "tag": args.tag, "n_adapters": len(names)},
                     sort_keys=True), flush=True)
    return run_dir
