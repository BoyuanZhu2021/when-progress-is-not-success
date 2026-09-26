# When Progress Is Not Success in Multi-Turn LLM Agent Red-Teaming

Code, run records and figure sources for the paper. It contains the training path that produced
every number in the paper (a trainable attacker LLM against a frozen, tool-using victim LLM), the
per-run records the paper's tables and figures were computed from, and the scripts that rebuild
those tables and figures.

Anonymous research release under double-blind review. Model weights are not included; the attacker
and victim are public Qwen checkpoints pinned by revision in `code/src/model_pins.py`.

## Layout

```
code/scripts/a3_multiturn_train.py   train and evaluate ONE (arm, seed) run
code/scripts/run_campaign.py         run arms x seeds across GPU lanes (seed-major, resumable)
code/scripts/serve_victim.py         start / probe / stop the vLLM victim server (fail-closed checks)
code/scripts/victim_lb.py            round-robin one trainee across several victim servers
code/scripts/a2_analyze.py           paired seed-level contrasts (paired t, seed bootstrap)
code/src/mt_grpo.py                  rewards, returns-to-go, state-stratified advantages, OFC weights
code/src/trace.py                    raw-trace logger (every generation is written with its prompt)
code/src/domains/agentdojo_*.py      goals, verifier, process score, multi-turn rollout loop
artifacts/<campaign>/<run>/          run_meta.json + progress.jsonl for every run used in the paper
artifacts/RUNS.md, MANIFEST.json     index of the records and their SHA-256 hashes
docs/results/sft-dynamics-4b-9b/     saved paired analyses of the mechanism study
paper/results/paper_data.json        per-run, per-read-out metrics + paired statistics (the paper's snapshot)
paper/scripts/make_figures.py        rebuilds the paper's PGFPlots figures and LaTeX tables from the snapshot
paper/figures/standalone.tex         compiles the result figures on their own
```

## Setting (as in the paper)

| | |
|---|---|
| Attacker | `Qwen/Qwen3.5-4B`, full-parameter bf16 fine-tuning (9B in the model-pairing study) |
| Victim | `Qwen/Qwen3.5-9B`, frozen, served by vLLM, executes the tools (`Qwen3.8-27B` FP8 in the pairing study) |
| Task | four-field data exfiltration built from AgentDojo banking / travel / workspace; 24 training goals, 24 held-out field combinations, 8 Slack transfer goals |
| Strict success | all four target values in one qualifying outbound call, decided by a programmatic verifier (no LLM judge) |
| Budget | at most 5 attacker turns, `G=6` rollouts per goal per iteration, 16 or 32 iterations |
| Optimisation | lr 2e-6, 8-bit AdamW, gradient-norm clip 1.0, sampling temperature 0.7, 112 generated tokens per response |
| Read-outs | 12 evaluation rollouts per goal at fixed iterations (every 2 in the original campaign, every 8 in the mechanism study) |

Arms (`--arm`): `sparse` (outcome-only group-relative RL, the reference), `dense` (process-progress
reward), `sft` (OFC: cross-entropy on every attacker response of every verified-successful
trajectory, no KL), `rl_pos` (sparse RL with negative advantages clipped at zero), `sft_all`
(cross-entropy on all trajectories). The original-campaign controls are `sparse` with extra flags:
lower threshold `--hindsight-m 2`, threshold curriculum `--m-curriculum`, adaptive allocation
`--adaptive-rollout`, success replay `--self-imitate N`, more rollouts `--G 12`. The exact settings
of every run are in its `run_meta.json` (indexed in `artifacts/RUNS.md`).

## Setup

Python 3.12. The runs used torch 2.8.0 (cu128), transformers 5.13.0, peft 0.19.1, bitsandbytes
0.49.2, agentdojo 0.1.35 and vLLM 0.24.0; see `requirements.txt`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install vllm==0.24.0            # only on the machine that serves the victim
python -c "import agentdojo, torch; print('ok')"
```

Model weights are downloaded from Hugging Face on first use (`HF_HOME` is respected). The revisions
used in the paper are pinned in `code/src/model_pins.py`; `serve_victim.py --plan` prints what it
will load.

## Serve the victim

One victim server per GPU that hosts a trainee. On Ampere (no FP8) the 9B victim is served in bf16;
`--quant fp8` is only accepted on capability 8.9+.

```bash
python code/scripts/serve_victim.py --plan  --model Qwen/Qwen3.5-9B --quant bf16 --gpu 0 --port 8001
python code/scripts/serve_victim.py --start --model Qwen/Qwen3.5-9B --quant bf16 --gpu 0 --port 8001 \
    --gpu-memory-utilization 0.42       # leaves room for a co-located 4B full-FT trainee on the same 80 GB card
python code/scripts/serve_victim.py --probe --model Qwen/Qwen3.5-9B --port 8001
```

`--probe` refuses to report success until the served model has returned a structured `tool_calls`
object; a victim whose tool calls are silently dropped produces a complete-looking run of zeros.
Runtime files (per-port manifest, pid, log) are written under `$OFC_RUNTIME_DIR` (default `./runtime`).

## Train one run

`--victim-model` must equal the served model name. The flags below are the paper's shared
configuration (`COMMON` in `run_campaign.py`); the arm and the seed are the only things that
change between runs of one campaign.

```bash
python code/scripts/a3_multiturn_train.py --arm sft --seed 2 \
    --run-dir artifacts/my_campaign/s2_sft --base-url http://127.0.0.1:8001/v1 \
    --victim-model Qwen/Qwen3.5-9B --steps 16 --eval-every 2 --concurrency 32 \
    --full-ft --baseline state_stratified --m-of-K 4 --kfield-K 4 --mechanism multidomain \
    --train-domains banking,travel,workspace --transfer-domain slack --n-train 8 --n-ood 8 \
    --data-seed 0 --T 5 --eval-k 12 --lr 2e-6 --beta-kl 0.02 --G 6
```

The run directory receives `run_meta.json`, `progress.jsonl` (one row per iteration, `eval` rows at
the read-outs: `ood_asr` is held-out strict ASR, `ood_phi` the held-out process score,
`transfer_asr` the Slack ASR), `turns.jsonl` (every generated attacker response with its prompt and
the victim's reply, per turn) and `eval_turns.jsonl`. `--ckpt-every N --resume` checkpoints
weights and optimiser state.

## Run a campaign

```bash
python code/scripts/run_campaign.py --run-root artifacts/my_campaign \
    --arms sparse,dense,sft --seeds 2-7 --lanes 0:1,1:1,2:1 \
    --base-url http://127.0.0.1:800{card}/v1 --victim-model Qwen/Qwen3.5-9B \
    --steps 16 --eval-every 2 --dry-run
```

Jobs are ordered seed-major so an interrupted campaign still yields complete matched seeds; a run is
skipped when its final training step and final evaluation are already on disk. Drop `--dry-run` to
launch. Measured on 80 GB A800 cards with the victim co-located: about 13 min per iteration for
the 4B attacker (rollout is roughly 90% of it), plus about 50 min per full evaluation.

## Analyse

```bash
python code/scripts/a2_analyze.py --runs artifacts/my_campaign/s*_sft artifacts/my_campaign/s*_sparse \
    --arms sft,sparse --metric ood_asr --steps 8 16
```

This is the paired seed-level contrast used throughout the paper (paired t-interval and a seed
bootstrap), plus the paired decay between the first and the last read-out.

## Rebuild the paper's figures and tables

```bash
pip install numpy
python paper/scripts/make_figures.py          # writes paper/figures/*.tex and paper/tables/*.tex
cd paper/figures && tectonic standalone.tex   # or pdflatex; renders the result figures
```

`paper/results/paper_data.json` is the snapshot every figure and table is computed from;
`paper/results/sources_manifest.json` lists the 199 record files it was exported from with their
SHA-256, and those files are the ones shipped under `artifacts/` and `docs/`.
`paper/scripts/prepare_results.py` re-exports the snapshot from run directories, but its audit
step reads raw `turns.jsonl` traces, which are not part of this release.

## Tests

The contract tests run on CPU and do not need model weights. Those that import `torch` or
`agentdojo` need the packages installed; the rest run with the standard library. Scripts under
`code/scripts/` run as files; modules under `code/src/` run with `python -m` from `code/`.

```bash
python code/scripts/run_campaign_test.py
python code/scripts/serve_victim_test.py
python code/scripts/victim_lb_test.py
python code/scripts/a2_analyze_test.py
python -c "import sys; sys.path.insert(0,'code'); from src.mt_grpo import _selftest; raise SystemExit(_selftest())"
cd code
python -m src.skillchain_rollout_test
python -m src.generation_runtime_test
python -m src.victim_decision_protocol_test
python -m src.domains.skillchain_env_test
python -m src.domains.dualtrack_env_test
python -m src.domains.agentdojo_multiturn_test      # needs agentdojo
python -m src.domains.agentdojo_infogated_test      # needs agentdojo
python scripts/a3_credit_methods_test.py           # needs agentdojo: reward / advantage / OFC weight rules
python scripts/a3_eval_trace_test.py               # needs agentdojo
python scripts/a3_checkpoint_test.py               # needs torch
```

## Run records

`artifacts/<campaign>/<run>/run_meta.json` and `progress.jsonl` are the files the paper's snapshot
was exported from; `artifacts/RUNS.md` indexes them and `artifacts/MANIFEST.json` records their
SHA-256. The only edit made for release is that the training machine's directory prefix was
dropped from the `serve_manifest_path` field of the mechanism-study and model-pairing records; the
manifest lists both the shipped hash and the original one for those files.

## Raw traces

Every generation of every run was logged with its full prompt (`code/src/trace.py`), and the
paper's metrics are derived from those traces. They contain complete attack conversations against
the victim, so they are not part of this public release; see `artifacts/RAW_TRACES.md`.

## Ethics

This code trains an attacker against a frozen victim inside benchmark-derived environments with
designated test data. It does not target production systems. The release is intended for the
evaluation of tool-agent security and for reproducing the paper; the raw attack conversations are
withheld for that reason.

## License

MIT (see `LICENSE`). AgentDojo and the Qwen models are subject to their own licenses.
