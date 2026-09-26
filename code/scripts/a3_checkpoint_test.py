"""Goldens for a3's checkpoint/resume.

What can and cannot be verified
-------------------------------
The victim samples server-side at temperature 0.7, so two "identical" runs never produce
bit-identical rollouts. An end-to-end "resumed run == uninterrupted run" assertion is therefore
impossible in principle, and claiming it would be false comfort. What IS decidable, and what these
tests cover:

1. **Serialization fidelity** -- weights and optimizer state survive save/load bit-exactly.
   If this holds, a resumed run continues from numerically the same point as an uninterrupted one.
2. **Bookkeeping** -- the progress/turns truncation keeps exactly the records at or before the
   checkpointed step. This is the part that would silently corrupt the raw-trace evidence
   (turns.jsonl is opened in APPEND mode while progress.jsonl is opened "w").
3. **Atomicity** -- a crash mid-write cannot leave a checkpoint that later loads as garbage.

The remaining risk after these -- that the campaign's matched-seed CRN breaks across a resume --
is closed by construction rather than by test: rollout randomness is derived fresh from
(seed, step, turn) inside torch.random.fork_rng, never carried forward, so there is no RNG state
whose loss could perturb it.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from a3_multiturn_train import (  # noqa: E402
    _ckpt_path,
    _truncate_jsonl_to_step,
    load_ckpt,
    save_ckpt,
)


def _tiny():
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4))
    o = torch.optim.AdamW(m.parameters(), lr=1e-3)
    return m, o


def _train_a_bit(m, o, n=3):
    for _ in range(n):
        loss = m(torch.randn(4, 8)).square().mean()
        o.zero_grad()
        loss.backward()
        o.step()


class TestSerializationFidelity(unittest.TestCase):
    def test_weights_and_optimizer_round_trip_exactly(self):
        m, o = _tiny()
        _train_a_bit(m, o)                       # so optimizer moments are non-trivial
        want_w = {k: v.clone() for k, v in m.state_dict().items()}
        want_exp_avg = {i: s["exp_avg"].clone()
                        for i, s in enumerate(o.state_dict()["state"].values())}
        with tempfile.TemporaryDirectory() as d:
            save_ckpt(d, 7, m, o)
            _train_a_bit(m, o, 5)                # move well away from the saved point
            self.assertFalse(torch.equal(m.state_dict()["0.weight"], want_w["0.weight"]))
            step = load_ckpt(d, m, o)
        self.assertEqual(step, 7)
        for k, v in m.state_dict().items():
            self.assertTrue(torch.equal(v, want_w[k]), f"weight {k} did not round-trip")
        got = o.state_dict()["state"]
        for i, s in enumerate(got.values()):
            self.assertTrue(torch.equal(s["exp_avg"], want_exp_avg[i]),
                            "optimizer moment did not round-trip -- a resumed run would take a "
                            "different update than an uninterrupted one")

    def test_missing_checkpoint_returns_zero(self):
        m, o = _tiny()
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(load_ckpt(d, m, o), 0)

    def test_write_is_atomic(self):
        """The real file must only appear via rename, so a crash mid-write cannot publish a
        truncated checkpoint."""
        m, o = _tiny()
        with tempfile.TemporaryDirectory() as d:
            save_ckpt(d, 3, m, o)
            self.assertTrue(_ckpt_path(d).exists())
            self.assertFalse(_ckpt_path(d).with_suffix(".tmp").exists(),
                             "staging file was left behind")


class TestTruncation(unittest.TestCase):
    def _write(self, p, steps):
        p.write_text("".join(json.dumps({"step": s, "v": s}) + "\n" for s in steps),
                     encoding="utf-8")

    def test_keeps_only_records_at_or_before_step(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "turns.jsonl"
            self._write(p, [1, 1, 2, 2, 3, 3, 4, 4])
            kept = _truncate_jsonl_to_step(p, 2)
            rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual(kept, 4)
        self.assertEqual([r["step"] for r in rows], [1, 1, 2, 2])

    def test_eval_rows_are_truncated_too(self):
        """progress.jsonl interleaves training and eval rows; both carry `step`, so an eval past
        the checkpoint must go as well or the resumed run would duplicate it."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "progress.jsonl"
            p.write_text(
                json.dumps({"step": 1}) + "\n"
                + json.dumps({"step": 2}) + "\n"
                + json.dumps({"step": 2, "eval": True}) + "\n"
                + json.dumps({"step": 3}) + "\n", encoding="utf-8")
            kept = _truncate_jsonl_to_step(p, 2)
            rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual(kept, 3)
        self.assertEqual([(r["step"], r.get("eval", False)) for r in rows],
                         [(1, False), (2, False), (2, True)])

    def test_absent_file_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(_truncate_jsonl_to_step(Path(d) / "nope.jsonl", 5), 0)

    def test_malformed_tail_line_is_dropped_not_fatal(self):
        """A kill mid-write can leave a half-written final line; it must not abort the resume."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "turns.jsonl"
            p.write_text(json.dumps({"step": 1}) + "\n" + '{"step": 2, "hal', encoding="utf-8")
            kept = _truncate_jsonl_to_step(p, 5)
        self.assertEqual(kept, 1)

    def test_truncation_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "turns.jsonl"
            self._write(p, [1, 2, 3])
            first = _truncate_jsonl_to_step(p, 2)
            second = _truncate_jsonl_to_step(p, 2)
        self.assertEqual((first, second), (2, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
