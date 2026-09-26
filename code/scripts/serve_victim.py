"""Start and verify the victim vLLM server for the CURRENT (a3) training path.

Why this exists
---------------
``a3_multiturn_train.py`` talks to the victim over ``--base-url`` (an OpenAI-compatible endpoint)
but the repo never committed the command that starts it -- every campaign launched it by hand.
Two production bugs came out of that gap, and both are reproduced here as fail-closed checks:

1. **Missing tool flags.** Without ``--enable-auto-tool-choice --tool-call-parser``, every victim
   call returns HTTP 400. Loud, caught quickly.
2. **Wrong parser.** With ``hermes`` against a Qwen3.5 victim, the server starts fine, answers
   200, and returns ``tool_calls: null`` forever -- because Qwen3.5 emits XML tool calls. The
   victim never acts, phi is identically 0, and the run completes and writes a full, correct-
   looking artifact of meaningless zeros. **This is the dangerous one**: it produces evidence-
   shaped garbage rather than an error.

So this script refuses to report success until it has seen the served model actually emit a
structured ``tool_calls`` object (``--probe``). Silence is not evidence.

Do NOT use ``h1_serve_victim_h20.py`` for this. That module implements a frozen formal-
verification contract for the archived single-GPU FP8 proof (``LEGACY_H20_PROFILE_ID``); its
``expected_cmdline()`` is hash-asserted by binding tests, is pinned to the 9B victim, and carries
neither tool flag. It is correct for what it verifies and wrong for training.

Usage
-----
    python code/scripts/serve_victim.py --plan  --model Qwen/Qwen3.5-27B --quant bf16 --tp 1
    python code/scripts/serve_victim.py --start --model Qwen/Qwen3.5-27B --quant fp8  --gpu 0
    python code/scripts/serve_victim.py --probe --model Qwen/Qwen3.5-27B
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

# Tool-call parser by model family. Deliberately a closed table with NO default: guessing a
# parser is what silently zeroed a campaign. An unknown family must stop the run and be looked
# up against the installed vLLM's --tool-call-parser choices, not defaulted to hermes.
TOOL_CALL_PARSERS = {
    # Both Qwen3.5 and Qwen3.8 report model_type "qwen3_5" / Qwen3_5ForConditionalGeneration and
    # emit XML tool calls. Qwen3.8 MUST be listed explicitly: "qwen3.8-27b".startswith("qwen3") is
    # True, so without this entry it would fall through to hermes -- the silent tool_calls=null
    # failure this whole module exists to prevent. Always confirm a new family with --probe.
    "qwen3.8": "qwen3_xml",   # verified by --probe on the box 2026-08-30
    "qwen3.5": "qwen3_xml",   # verified 2026-08: Qwen3.5 emits XML tool calls, NOT hermes
    "qwen3": "hermes",
    "qwen2.5": "hermes",
}

FP8_MIN_CAPABILITY = (8, 9)   # Ada/Hopper. Ampere (A100/A800 = 8.0) has no FP8.

DEFAULT_PORT = 8000
DEFAULT_HOST = "127.0.0.1"
# Runtime files (manifest, pid, log) live under $OFC_RUNTIME_DIR, default ./runtime.
RUNTIME_DIR = os.environ.get("OFC_RUNTIME_DIR", "runtime")
Path(RUNTIME_DIR).mkdir(parents=True, exist_ok=True)
DEFAULT_MANIFEST = f"{RUNTIME_DIR}/victim_serve.json"


def canonical_manifest(port: int) -> str:
    """Per-PORT manifest path, always written by --start regardless of --manifest.

    A fixed default cannot describe a box serving several victims at once: whichever server wrote
    it last wins, and every other lane's run_meta then names the wrong model. G0 recorded
    `Qwen3.8-27B-FP8` for runs that actually talked to a 9B for exactly this reason. Keyed by port
    because the port is the only thing the trainer knows about its victim (`--base-url`).
    """
    return f"{RUNTIME_DIR}/victim_serve_{int(port)}.json"
DEFAULT_HF_HOME = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))


class PreflightError(RuntimeError):
    """A condition that would produce silent, evidence-shaped garbage if allowed to proceed."""


def model_family(model: str) -> str:
    """``Qwen/Qwen3.5-27B`` -> ``qwen3.5``. Longest matching key wins so qwen3.5 beats qwen3."""
    name = model.split("/")[-1].lower()
    hits = [k for k in TOOL_CALL_PARSERS if name.startswith(k)]
    if not hits:
        raise PreflightError(
            f"no tool-call parser known for {model!r}.\n"
            f"  Known families: {sorted(TOOL_CALL_PARSERS)}\n"
            f"  Refusing to guess -- a wrong parser returns tool_calls=null silently and the\n"
            f"  run completes with phi identically 0. Check the installed vLLM's\n"
            f"  `--tool-call-parser` choices, confirm with --probe, then add the family here."
        )
    return max(hits, key=len)


def parser_for_model(model: str) -> str:
    return TOOL_CALL_PARSERS[model_family(model)]


def compute_capability(gpu: int = 0) -> tuple[int, int] | None:
    """(major, minor) for the given GPU, or None if it cannot be determined."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={gpu}", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True).stdout.strip()
        major, _, minor = out.partition(".")
        return int(major), int(minor)
    except Exception:
        return None


def validate_quant(quant: str, capability: tuple[int, int] | None,
                   model: str | None = None) -> None:
    """FP8 on pre-Ada silently degrades or fails to load; refuse it up front.

    ``model`` is inspected too: an ALREADY-quantized checkpoint (e.g. ``Qwen3.8-27B-FP8``) needs
    the same hardware support even when ``--quant auto`` leaves the flag off, so the guard must
    not be bypassable by simply not naming the precision.
    """
    if quant not in ("bf16", "fp8", "auto"):
        raise PreflightError(f"unsupported --quant {quant!r} (use bf16, fp8, or auto)")
    wants_fp8 = quant == "fp8" or (model is not None and "fp8" in model.lower())
    if wants_fp8 and capability is not None and capability < FP8_MIN_CAPABILITY:
        raise PreflightError(
            f"FP8 weights requested ({quant!r}, model={model!r}) but this GPU is "
            f"compute capability "
            f"{capability[0]}.{capability[1]} "
            f"(needs >= {FP8_MIN_CAPABILITY[0]}.{FP8_MIN_CAPABILITY[1]}).\n"
            f"  Ampere (A100/A800, sm80) has NO FP8 -- serve bf16 instead. Hopper (H20/H100,\n"
            f"  sm90) and Ada do support it."
        )


def build_serve_argv(model: str, *, revision: str | None = None, quant: str = "bf16",
                     tp: int = 1, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                     gpu_memory_utilization: float = 0.90, max_model_len: int = 8192,
                     max_num_seqs: int = 256, served_name: str | None = None,
                     tool_call_parser: str | None = None, dtype: str | None = None,
                     python: str = "python") -> list[str]:
    """Canonical argv for the training victim. The two tool flags are NOT optional."""
    parser = tool_call_parser or parser_for_model(model)
    # With a pre-quantized checkpoint let the checkpoint decide BOTH quantization and dtype;
    # forcing --dtype bfloat16 over a repo that declares its own torch_dtype can conflict.
    if dtype is None:
        dtype = "auto" if quant == "auto" else "bfloat16"
    argv = [
        python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--served-model-name", served_name or model,
        "--host", host,
        "--port", str(port),
        "--dtype", dtype,
        "--tensor-parallel-size", str(tp),
        "--gpu-memory-utilization", f"{gpu_memory_utilization:g}",
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
        # ---- the two flags whose absence 400s every call ----
        "--enable-auto-tool-choice",
        "--tool-call-parser", parser,
    ]
    if revision:
        argv.extend(["--revision", revision])
    if quant == "fp8":
        argv.extend(["--quantization", "fp8"])
    return argv


PROBE_TOOL = [{
    "type": "function",
    "function": {
        "name": "get_balance",
        "description": "Return the account balance.",
        "parameters": {"type": "object",
                       "properties": {"account_id": {"type": "string"}},
                       "required": ["account_id"]},
    },
}]


def probe_tool_calls(base_url: str, model: str, timeout: int = 120) -> dict:
    """Assert the server emits STRUCTURED tool_calls.

    This is the check that catches a wrong parser, which otherwise degrades silently to phi==0
    for an entire campaign.
    """
    import openai
    client = openai.OpenAI(api_key="EMPTY", base_url=base_url, timeout=timeout)
    resp = client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=256, tools=PROBE_TOOL, tool_choice="auto",
        messages=[{"role": "user",
                   "content": "What is the balance of account ACC-1? Use the tool."}],
    )
    msg = resp.choices[0].message
    calls = msg.tool_calls or []
    result = {
        "ok": bool(calls),
        "n_tool_calls": len(calls),
        "tool_names": [c.function.name for c in calls],
        "content_preview": (msg.content or "")[:300],
    }
    if not calls:
        raise PreflightError(
            "server returned NO structured tool_calls.\n"
            f"  content was: {result['content_preview']!r}\n"
            "  The model probably emitted a tool call the parser could not read (e.g. Qwen3.5\n"
            "  XML parsed by `hermes`). Training would run to completion with phi identically 0\n"
            "  and write a full artifact of meaningless zeros. Fix the --tool-call-parser."
        )
    return result


def resolve_local_snapshot(model: str, revision: str | None = None,
                           hf_home: str = DEFAULT_HF_HOME) -> str | None:
    """On-disk snapshot dir for a cached repo id, or None if it is not cached.

    Verified on this box 2026-08-29: ``transformers`` 5.13 cannot resolve a repo id from the HF
    cache while offline -- it fails in BOTH conda envs, *including the one that performed the
    download*, and the box has no direct route to huggingface.co without ``network_turbo``.
    Loading the same weights by snapshot path succeeds. So vLLM is pointed at the directory, while
    ``--served-model-name`` keeps the logical repo id on the API, which is what the trainer sends.

    Ambiguity is refused rather than guessed: two cached revisions with no ``--revision`` given
    would otherwise silently pick one, and "which weights actually ran" is not a coin flip.
    """
    base = Path(hf_home) / "hub" / ("models--" + model.replace("/", "--")) / "snapshots"
    if not base.is_dir():
        return None
    if revision:
        p = base / revision
        return str(p) if (p / "config.json").is_file() else None
    cands = sorted(d for d in base.iterdir() if (d / "config.json").is_file())
    if not cands:
        return None
    if len(cands) > 1:
        raise PreflightError(
            f"{model} has {len(cands)} cached revisions and no --revision was given:\n  "
            + "\n  ".join(d.name for d in cands)
            + "\n  Refusing to guess which weights to serve. Pass --revision."
        )
    return str(cands[0])


def launch_command(argv: list[str], env: dict[str, str]) -> str:
    """Shell string that applies ``env`` and runs ``argv``, safe to prefix with ``nohup``.

    A bare ``VAR=x cmd`` prefix is shell syntax, not a command: ``nohup`` execs its first argument
    and dies with "failed to run command 'CUDA_VISIBLE_DEVICES=1'". Going through ``env`` keeps
    the first token a real executable, so the same string works bare, under nohup, or under any
    other exec-style wrapper.
    """
    return "env " + " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " " + shlex.join(argv)


def _manifest_for(base_url: str, manifest_path: str | None) -> list[str]:
    """Manifest paths to try.

    An EXPLICIT ``manifest_path`` is used alone: falling back from a caller-named file to an
    unrelated one is the exact failure this function exists to prevent (a pre-existing golden,
    ``test_absence_is_explicit_not_omitted``, caught a first version that did fall back — on the box
    it silently resolved a missing path to the previous campaign's 27B manifest).

    With no explicit path, try the per-port file this serve wrote, then the legacy fixed default so
    runs recorded before the per-port scheme still resolve."""
    if manifest_path:
        return [manifest_path]
    cands: list[str] = []
    try:
        port = int(str(base_url).split(":")[2].split("/")[0])
        cands.append(canonical_manifest(port))
    except Exception:
        pass
    cands.append(DEFAULT_MANIFEST)
    return cands


def victim_identity(base_url: str, manifest_path: str | None = None,
                    timeout: int = 10) -> dict:
    """Best-effort record of HOW the victim was served, for ``run_meta.json`` (§5.3).

    The full-FT campaign's serve command became unrecoverable the moment its box was destroyed,
    because ``run_meta`` captured the training config only. Victim precision is a CONTROLLED
    variable (bf16-vs-fp8 was deviation D1), so it belongs in the run's self-description.

    Never raises: a metadata gap must not kill a training run. Absence is recorded explicitly as
    ``null`` rather than omitted, so a reader can tell "not served by our launcher" apart from
    "nobody thought to write it down".
    """
    out: dict = {"base_url": base_url, "serve_manifest": None, "served_models": None,
                 "serve_manifest_path": None}
    for cand in _manifest_for(base_url, manifest_path):
        try:
            m = json.loads(Path(cand).read_text(encoding="utf-8"))
        except Exception:
            continue
        out["serve_manifest"] = {k: m.get(k) for k in (
            "model", "revision", "quant", "tp", "tool_call_parser", "compute_capability")}
        out["serve_manifest_path"] = cand          # which file this block came from
        break
    try:
        import openai
        # max_retries=0: this is best-effort metadata at run start, not a health check.
        # The default retry ladder would stall every run by ~30s against a dead endpoint.
        cli = openai.OpenAI(api_key="EMPTY", base_url=base_url, timeout=timeout,
                            max_retries=0)
        out["served_models"] = [d.id for d in cli.models.list().data]
    except Exception:
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="print the serve command, run nothing")
    mode.add_argument("--start", action="store_true", help="start the server detached")
    mode.add_argument("--probe", action="store_true",
                      help="verify a running server emits tool_calls")
    mode.add_argument("--stop", action="store_true",
                      help="stop the server recorded in --pid-file (and its children)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--quant", default="bf16", choices=("bf16", "fp8", "auto"),
                    help="auto = omit --quantization and let vLLM read the checkpoint "
                         "(correct for pre-quantized repos like Qwen3.8-27B-FP8)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0, help="first CUDA device for the server")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    # Qwen3.8 is a HYBRID Mamba/attention model: every concurrent decode needs its own Mamba
    # cache block, and vLLM refuses to start when max_num_seqs exceeds the available blocks
    # (256 > 137 at util 0.40). a3 drives --concurrency 16, so 64 is 4x real demand and still
    # far under the block budget. The pure-attention 9B victim never hit this.
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--flashinfer-fp8-gemm", action="store_true",
                    help="re-enable VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER (needs TRT-LLM cubins; "
                         "without them engine init dies in fp8_blockscale_gemm_sm90)")
    ap.add_argument("--tool-call-parser", default=None, help="override the family table")
    ap.add_argument("--base-url", default=None, help="for --probe (default from --host/--port)")
    ap.add_argument("--hf-home", default=DEFAULT_HF_HOME)
    ap.add_argument("--pid-file", default=f"{RUNTIME_DIR}/victim_serve.pid")
    ap.add_argument("--no-local-resolve", action="store_true",
                    help="pass the repo id to vLLM instead of the cached snapshot dir")
    ap.add_argument("--log", default=f"{RUNTIME_DIR}/victim_serve.log")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = ap.parse_args()

    base_url = args.base_url or f"http://{args.host}:{args.port}/v1"

    if args.stop:
        pf = Path(args.pid_file)
        if not pf.exists():
            print(f"no pid file at {pf}; nothing recorded to stop", file=sys.stderr)
            return 1
        pid = pf.read_text(encoding="utf-8").strip()
        # pkill -P is scoped by PARENT PID, not a command pattern: `pkill -f` self-matches its
        # own wrapper and has caused real damage on this project.
        subprocess.run(f"pkill -P {pid}; kill {pid}", shell=True, executable="/bin/bash")
        subprocess.run("sleep 6", shell=True, executable="/bin/bash")
        left = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip()
        pf.unlink(missing_ok=True)
        print(f"stopped pid {pid}")
        print(f"remaining GPU compute apps: {left or 'none'}")
        return 0

    if args.probe:
        if not args.model:
            ap.error("--probe needs --model (the served-model-name)")
        try:
            print(json.dumps(probe_tool_calls(base_url, args.model), indent=2))
        except PreflightError as e:
            print(f"PROBE FAILED: {e}", file=sys.stderr)
            return 1
        print("\nOK: server emits structured tool_calls.")
        return 0

    if not args.model:
        ap.error("--plan/--start need --model")
    cap = compute_capability(args.gpu)
    try:
        validate_quant(args.quant, cap, args.model)
        # Point vLLM at the snapshot dir when the repo is cached: transformers cannot resolve a
        # repo id from the cache offline on this box. --served-model-name keeps the repo id.
        local = None if args.no_local_resolve else resolve_local_snapshot(
            args.model, args.revision, args.hf_home)
        # Resolve the parser from the REPO ID, never from the substituted path: the tool-call
        # format is a property of the model's identity, and a filesystem path has no family.
        parser = args.tool_call_parser or parser_for_model(args.model)
        argv = build_serve_argv(
            local or args.model, revision=None if local else args.revision, quant=args.quant,
            tp=args.tp, host=args.host, port=args.port,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
            served_name=args.model,
            tool_call_parser=parser, python=sys.executable)
    except PreflightError as e:
        print(f"PREFLIGHT FAILED: {e}", file=sys.stderr)
        return 1

    devices = ",".join(str(args.gpu + i) for i in range(args.tp))
    # vLLM shells out to `ninja` to JIT-compile kernels. Invoking the env's python by ABSOLUTE
    # path does not put that env's bin/ on PATH, so ninja is not found and engine init dies with
    # FileNotFoundError. Prepend the interpreter's own bin dir so its helper binaries resolve.
    env_bin = str(Path(sys.executable).parent)
    env = {"CUDA_VISIBLE_DEVICES": devices, "HF_HUB_DISABLE_XET": "1",
           "HF_HOME": args.hf_home, "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
           "PATH": env_bin + os.pathsep + os.environ.get("PATH", "")}
    # FlashInfer downloads precompiled cubins at RUNTIME (e.g. fp8_blockscale_gemm_sm90). Without
    # a route out, engine init dies with `Assertion failed: !cubin.empty() || isPathValid(path_)`
    # -- which reads like a corrupt cache, not a network problem. This box only reaches the
    # internet through a proxy, so carry it into the server's environment. Verified: the cubin
    # repository answers 200 through the proxy and is unreachable without it.
    for var in ("http_proxy", "https_proxy", "no_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                "NO_PROXY", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"):
        if os.environ.get(var):
            env[var] = os.environ[var]
    # vLLM turns on FlashInfer's FP8 block-scale GEMM by default on SM90+, but that path needs
    # TensorRT-LLM cubins this install does not ship. Serving an FP8 checkpoint then dies in
    # fp8_blockscale_gemm_sm90 with `Assertion failed: !cubin.empty() || isPathValid(path_)` --
    # a message that reads like a corrupt cache rather than a missing optional kernel. Turning
    # it off selects a different GEMM implementation; correctness is unaffected, and any
    # throughput difference is a controlled variable shared by every arm.
    if not args.flashinfer_fp8_gemm:
        env["VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER"] = "0"
    cmd = launch_command(argv, env)

    print(f"# compute capability : {cap[0]}.{cap[1]}" if cap else "# compute capability : UNKNOWN")
    print(f"# tool-call parser   : {args.tool_call_parser or parser_for_model(args.model)}")
    print(f"# weights            : {local or args.model}" + ("" if local else "  (repo id -- NOT resolved to a local snapshot)"))
    print(f"# GPUs               : {devices}")
    print(cmd)

    if args.plan:
        print("\n# --plan only; nothing started. Verify AFTER starting with:")
        print(f"#   python {Path(__file__).name} --probe --model {args.model} "
              f"--base-url {base_url}")
        return 0

    manifest_body = json.dumps({
        "model": args.model, "revision": args.revision, "quant": args.quant, "tp": args.tp,
        "tool_call_parser": args.tool_call_parser or parser_for_model(args.model),
        "compute_capability": f"{cap[0]}.{cap[1]}" if cap else None,
        "weights_path": local, "max_num_seqs": args.max_num_seqs,
        "base_url": base_url, "pid_file": args.pid_file,
        "argv": argv, "env": env,
    }, indent=2)
    # Write BOTH the caller's path and the canonical per-port one, ALWAYS overwriting. The trainer
    # only knows its victim by --base-url, so the per-port file is the one it can actually find;
    # without it, a box serving four victims leaves three runs describing the wrong model (G0 did).
    for dest in {args.manifest, canonical_manifest(args.port)}:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_text(manifest_body, encoding="utf-8")
    # Record the detached pid. Killing whatever launched this script does NOT kill the server
    # (it is nohup'd), and a grep-based hunt is how a live server got its weights deleted out
    # from under it. A pid file makes teardown deterministic: see --stop.
    subprocess.Popen(
        f"nohup {cmd} > {shlex.quote(args.log)} 2>&1 & echo $! > {shlex.quote(args.pid_file)}",
        shell=True, executable="/bin/bash")
    print(f"\nstarted detached -> {args.log}\nmanifest -> {args.manifest}")
    print(f"WAIT for load, then you MUST verify:\n"
          f"  python {Path(__file__).name} --probe --model {args.model} --base-url {base_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
