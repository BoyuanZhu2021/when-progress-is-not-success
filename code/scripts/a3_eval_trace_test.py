"""Goldens for §5.3 eval tracing in `evaluate_asr` (added 2026-09-02).

Before this change the primary DV (OOD ASR) was computed from 2x672 eval episodes per run whose raw
text was never written anywhere; `turns.jsonl` held exactly the 16x144 TRAINING generations and nothing
else. These tests pin the properties the fix must have:

  * every eval episode lands in `eval_turns.jsonl` with its decoded prompt + attacker raw text +
    victim raw text, tagged with split/step/arm/seed so it is attributable;
  * it is a SEPARATE stream -- `turns.jsonl` is untouched, so the training-record count that
    `sync_artifacts.py --verify` checks against `run_meta.json` stays exact;
  * tracing does not change the readout (same tuple with and without a tracer);
  * `trace=None` is a no-op, so every pre-existing call site keeps its old behaviour;
  * the record format is the ONE shared helper the training loop also uses;
  * the tracer is fail-closed on a record with no text.

The rollout is stubbed: this is about what gets written, not about the victim.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import a3_multiturn_train as a3  # noqa: E402
from src.trace import TraceLogger  # noqa: E402


class FakeTok:
    def decode(self, ids, skip_special_tokens=False):
        return "P:" + ",".join(str(i) for i in ids)


def fake_rollout(specs, gen, **kw):
    """Deterministic episodes: goal g, K-th repeat k -> success iff k is even; 2 turns each."""
    out = []
    for n, goal in enumerate(specs):
        k = n % 2
        out.append({
            "goal_id": goal,
            "turns": [
                {"turn": 1, "response": f"atk-{goal}-{k}-t1", "phi": 0.25, "phi_true": 0.25,
                 "prompt_ids": [1, 2, 3], "resp_ids": [9], "victim_reply": f"vic-{goal}-{k}-t1"},
                {"turn": 2, "response": f"atk-{goal}-{k}-t2", "phi": 1.0 if k == 0 else 0.5,
                 "phi_true": 1.0 if k == 0 else 0.5,
                 "prompt_ids": [4, 5], "resp_ids": [8], "victim_reply": f"vic-{goal}-{k}-t2"},
            ],
            "phi_trace": [0.25, 1.0 if k == 0 else 0.5],
            "count_trace": [0.25, 1.0 if k == 0 else 0.5],
            "success": k == 0,
            "max_phi": 1.0 if k == 0 else 0.5,
            "security_count": 1.0 if k == 0 else 0.5,
        })
    return out


COMMON = dict(suite=None, T=2, K_eval=2, client=None, victim_model="v", victim_sys="s",
              concurrency=1, chunk=4, G=6, tau_m=1.0, K_fields=4)
GOALS = ["g0", "g1", "g2"]          # 3 goals x K_eval 2 = 6 episodes, chunk 4 -> two waves


class TestEvalTracing(unittest.TestCase):
    def _run(self, trace, meta=None):
        try:
            with mock.patch.object(a3, "multiturn_rollout_batch", fake_rollout):
                return a3.evaluate_asr(GOALS, gen=None, trace=trace, tok=FakeTok(), trace_meta=meta, **COMMON)
        finally:
            if trace is not None:
                trace.close()          # TraceLogger keeps its append handle open; Windows won't rmdir past it

    def test_every_eval_episode_is_written_with_prompt_and_raw_text(self):
        with tempfile.TemporaryDirectory() as td:
            tr = TraceLogger(td)
            self._run(tr, {"step": 16, "arm": "dense", "seed": 4, "split": "ood"})
            recs = [json.loads(l) for l in (Path(td) / "eval_turns.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(recs), 6, "3 goals x K_eval 2, across two waves")
            for r in recs:
                self.assertEqual(r["kind"], "eval_generation")
                self.assertEqual((r["step"], r["arm"], r["seed"], r["split"]), (16, "dense", 4, "ood"))
                self.assertIn(r["goal"], GOALS)
                turns = r["response"]["turns"]
                self.assertEqual(len(turns), 2)
                for t in turns:
                    self.assertTrue(t["prompt"].startswith("P:"), "prompt must be the DECODED text")
                    self.assertTrue(t["attacker"].startswith("atk-"), "attacker raw text")
                    self.assertTrue(t["victim"].startswith("vic-"), "victim raw text")
                self.assertEqual(r["phi_trace"], r["count_trace"])
            self.assertEqual(sorted(r["goal"] for r in recs), sorted(GOALS * 2))

    def test_eval_never_touches_the_training_stream(self):
        with tempfile.TemporaryDirectory() as td:
            self._run(TraceLogger(td), {"split": "ood"})
            self.assertFalse((Path(td) / "turns.jsonl").exists(),
                             "turns.jsonl must stay steps x goals x G; eval has its own file")

    def test_tracing_does_not_change_the_readout(self):
        with tempfile.TemporaryDirectory() as td:
            with_trace = self._run(TraceLogger(td), {"split": "ood"})
        without = self._run(None)
        self.assertEqual(with_trace, without)
        asr, phi, z, phi_true, count, by_m = without
        self.assertEqual(asr, 0.5, "each goal: 1 of 2 repeats succeeds")
        # the failing repeat peaks at count 0.5: clears m1 (0.25) and m2 (0.5), not m3 (0.75) or m4 (1.0)
        self.assertEqual(by_m, {"m1": 1.0, "m2": 1.0, "m3": 0.5, "m4": 0.5})

    def test_trace_none_writes_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(a3, "multiturn_rollout_batch", fake_rollout):
                a3.evaluate_asr(GOALS, gen=None, **COMMON)          # the pre-2026-09-02 call shape
            self.assertEqual(list(Path(td).iterdir()), [])

    def test_training_and_eval_share_one_record_format(self):
        """The training loop calls episode_trace_record too; the eval record must be that plus meta."""
        ep = fake_rollout(["g0"], None)[0]
        base = a3.episode_trace_record(ep, FakeTok())
        self.assertEqual(set(base), {"goal", "phi_trace", "count_trace", "success", "response"})
        with tempfile.TemporaryDirectory() as td:
            self._run(TraceLogger(td), {"split": "ood", "step": 8})
            rec = json.loads((Path(td) / "eval_turns.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(set(rec) - {"ts", "kind", "split", "step"}, set(base))
        self.assertEqual(rec["response"]["turns"][0]["prompt"], "P:1,2,3")

    def test_prompt_falls_back_to_none_without_a_tokenizer_but_text_is_kept(self):
        ep = fake_rollout(["g0"], None)[0]
        rec = a3.episode_trace_record(ep, None)
        self.assertIsNone(rec["response"]["turns"][0]["prompt"])
        self.assertEqual(rec["response"]["turns"][0]["attacker"], "atk-g0-0-t1")

    def test_log_eval_turn_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            with TraceLogger(td) as tr, self.assertRaises(ValueError):
                tr.log_eval_turn({"split": "ood", "goal": "g0"})
            self.assertFalse((Path(td) / "eval_turns.jsonl").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
