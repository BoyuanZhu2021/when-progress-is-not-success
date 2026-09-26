"""Append-only experiment trace. Records EVERY model generation (full input + output)
and every episode-level event, so the whole run is auditable / re-analyzable offline.

JSONL streams per run dir (flushed after each write, thread-safe, opened on first use
so a run only materializes the streams it actually uses):
  - calls.jsonl   one record per API LLM call: role, context, request params, full
                  messages (input), content + reasoning (output), usage, latency.
  - turns.jsonl   one record per local-generation decision: full prompt messages and
                  the RAW generated text, plus whatever the caller attaches (action,
                  reward, advantage). This is the local-inference counterpart of
                  calls.jsonl -- an in-process HF/vLLM policy never goes through
                  `llm_client`, so without it the raw responses are lost.
  - events.jsonl  one record per episode-level event: episode start, per-turn
                  summary, oracle verdict, episode result.
  - records.jsonl one per-episode analysis record.

Plus run_meta.json (config snapshot + repro metadata) via `write_meta`.

`LoggedClient` wraps llm_client.chat so a call is recorded even if it later raises.
`llm_client` is imported lazily by `LoggedClient` only, so GPU trainers can use
`TraceLogger` without pulling in the API-client dependency chain.
"""
from __future__ import annotations

import datetime
import json
import threading
import time
from pathlib import Path


def _now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="milliseconds")


class TraceLogger:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._streams: dict[str, object] = {}
        self.n_calls = 0
        self.n_events = 0
        self.n_turns = 0

    def _stream(self, name: str):
        """Open ``<name>.jsonl`` on first use (caller already holds the lock)."""
        fh = self._streams.get(name)
        if fh is None:
            fh = open(self.run_dir / f"{name}.jsonl", "a", encoding="utf-8")
            self._streams[name] = fh
        return fh

    def _write(self, name: str, rec: dict):
        with self._lock:
            fh = self._stream(name)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

    def log_call(self, rec: dict):
        self._write("calls", {"ts": _now_iso(), "kind": "llm_call", **rec})
        self.n_calls += 1

    def log_turn(self, rec: dict):
        """One local-generation decision, including the RAW response text.

        Fail-closed: a turn record without its raw response is a silent evidence hole,
        so refuse to write one rather than persist an unauditable trace.
        """
        if "response" not in rec:
            raise ValueError("turn record must carry the raw 'response' text")
        self._write("turns", {"ts": _now_iso(), "kind": "generation", **rec})
        self.n_turns += 1

    def log_eval_turn(self, rec: dict):
        """One EVAL episode, raw text included, to ``eval_turns.jsonl``.

        A SEPARATE stream from ``turns.jsonl`` so the training-record count (steps x goals x G) that
        ``sync_artifacts.py`` verifies against ``run_meta.json`` stays exact. Same fail-closed rule as
        ``log_turn``: the DV is computed from these episodes, and a record without its text is an
        evidence hole. Before 2026-09-02 nothing wrote them at all.
        """
        if "response" not in rec:
            raise ValueError("eval turn record must carry the raw 'response' text")
        self._write("eval_turns", {"ts": _now_iso(), "kind": "eval_generation", **rec})
        self.n_turns += 1

    def log_event(self, rec: dict):
        self._write("events", {"ts": _now_iso(), **rec})
        self.n_events += 1

    def log_record(self, rec: dict):
        """One per-episode analysis record, written crash-safely as it completes."""
        self._write("records", rec)

    def write_meta(self, meta: dict):
        """Config + repro snapshot, so a run dir is self-describing on its own."""
        (self.run_dir / "run_meta.json").write_text(
            json.dumps({"written_at": _now_iso(), **meta}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def close(self):
        with self._lock:
            for fh in self._streams.values():
                fh.close()
            self._streams.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


_REQ_KEYS = (
    "provider", "model", "max_tokens", "temperature", "enable_thinking", "seed", "extra",
)


class LoggedClient:
    """Every .chat() is logged to the TraceLogger with full input + output.

    `role` (e.g. 'attacker', 'target') and `context` (episode_id, turn, etc.) are
    attached for slicing during analysis.
    """

    def __init__(self, trace: TraceLogger):
        self.trace = trace

    def chat(self, *, role: str, context: dict | None = None, **kwargs) -> dict:
        try:
            from . import llm_client
        except ImportError:  # imported as a top-level module (src on sys.path)
            import llm_client
        t0 = time.time()
        rec: dict = {
            "role": role,
            "context": context or {},
            "request": {k: kwargs.get(k) for k in _REQ_KEYS},
            "messages": kwargs.get("messages"),
        }
        try:
            r = llm_client.chat(**kwargs)
            rec["ok"] = True
            rec["response"] = {
                "content": r["content"],
                "reasoning": r["reasoning"],
                "usage": r["usage"],
                "finish_reason": r.get("finish_reason"),
                "latency": r["latency"],
            }
            return r
        except Exception as e:  # noqa: BLE001
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["elapsed"] = time.time() - t0
            raise
        finally:
            self.trace.log_call(rec)
