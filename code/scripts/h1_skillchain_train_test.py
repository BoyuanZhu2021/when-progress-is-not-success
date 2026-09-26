"""CPU end-to-end test for the skill-chain GRPO trainer.

Runs the REAL update loop (token_logps_batch, disable_adapter reference pass, token-normalized
PG+KL, grad clip, AdamW step) with a tiny torch model that mimics the peft LoRA interface. No GPU,
no transformers/peft. Also checks the pipeline reproduces Claim 1: dense harvests more
nonzero-advantage signal than sparse when full success is rare but partial progress is common.
"""
from __future__ import annotations

import contextlib
import random
import re
import sys
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import h1_skillchain_train as trainer  # noqa: E402
from src.domains.skillchain_env import hop_transform  # noqa: E402


class TinyLoRALM(nn.Module):
    """Frozen base projection + a rank-r LoRA delta (trainable, B init 0), with a peft-style
    ``disable_adapter`` context. Enough for token_logps_batch and the update to run for real."""

    def __init__(self, vocab=32, dim=8, rank=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.embed.weight.requires_grad_(False)
        self.base = nn.Parameter(torch.randn(dim, vocab) * 0.1, requires_grad=False)
        self.lora_A = nn.Parameter(torch.randn(rank, dim) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(vocab, rank))
        self._adapter_on = True

    def forward(self, input_ids, attention_mask=None):
        h = self.embed(input_ids)
        logits = h @ self.base
        if self._adapter_on:
            logits = logits + h @ (self.lora_B @ self.lora_A).t()
        return types.SimpleNamespace(logits=logits)

    @contextlib.contextmanager
    def disable_adapter(self):
        prev = self._adapter_on
        self._adapter_on = False
        try:
            yield
        finally:
            self._adapter_on = prev


def make_gen(recall_prob, seed, vocab=32):
    """Mock generator with a SHARED RNG advanced per item, i.e. temperature-like sampling: the G
    rollouts of one goal start from an identical transcript but diverge (as a real sampler would),
    which is what gives GRPO its intra-group variance. A fresh generator with the same seed
    reproduces the identical rollout sequence, so dense and sparse still see identical Φ."""
    rng = random.Random(seed)

    def gen(batch_messages):
        out = []
        for messages in batch_messages:
            obs = messages[-1]["content"]
            tool = re.search(r"Call tool `([^`]+)`", obs).group(1)
            cands = re.search(r"Candidate tokens: (.+)", obs).group(1).split()
            source = None
            for m in messages:
                hit = re.search(rf"\({re.escape(tool)}\):\s*(\S+)", m["content"])
                if hit:
                    source = hit.group(1)
            correct = hop_transform(source) if source is not None else None
            token = correct if (correct and rng.random() < recall_prob) else rng.choice(cands)
            ids = [1 + rng.randrange(vocab - 2)]
            out.append({
                "text": f"ACTION {tool} {token}",
                "prompt_ids": torch.tensor(ids + [2, 3], dtype=torch.long),
                "resp_ids": torch.tensor([4, 5], dtype=torch.long),
            })
        return out
    return gen


class SkillchainTrainCpuE2E(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.goals = trainer.build_goals(m=3, n_distractors=7, n_train=8, n_cal=4, n_ood=4, seed=0)
        self.schedule = trainer.build_schedule(n_train=8, n_goals_per_step=4, G=6, steps=3, seed=0)

    def _run(self, arm, tmp):
        model = TinyLoRALM()
        rows = trainer.run_skillchain_training(
            model=model, generator=make_gen(0.35, seed=7), goals=self.goals["train"],
            schedule=self.schedule, arm=arm, steps=3, T=4, tau=1.0, pad_token_id=0,
            run_dir=Path(tmp) / arm, seed=0,
        )
        return model, rows

    def test_update_runs_and_moves_lora(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            model, rows = self._run("dense", tmp)
        self.assertEqual(len(rows), 3)
        self.assertGreater(rows[-1]["lora_l2_delta"], 0.0)          # the policy actually updated
        for r in rows:
            self.assertTrue(all(k in r for k in (
                "mean_max_phi", "success_rate", "n_nonzero_advantage",
                "all_zero_group_fraction", "grad_norm", "pg_loss", "kl_loss")))
            self.assertTrue(r["grad_norm"] == 0.0 or r["grad_norm"] == r["grad_norm"])  # finite

    def test_dense_harvests_more_signal_than_sparse(self):
        """Claim 1 through the full pipeline: identical rollouts (shared generator seed), only the
        reward differs, and dense gets strictly more nonzero-advantage decisions."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            _dm, dense = self._run("dense", tmp)
            _sm, sparse = self._run("sparse", tmp)
        dense_nonzero = sum(r["n_nonzero_advantage"] for r in dense)
        sparse_nonzero = sum(r["n_nonzero_advantage"] for r in sparse)
        self.assertGreater(dense_nonzero, sparse_nonzero)
        # sparse starves: with full success rare, most sparse groups are all-zero
        self.assertGreaterEqual(
            sum(r["all_zero_group_fraction"] for r in sparse),
            sum(r["all_zero_group_fraction"] for r in dense))

    def test_phi_traces_identical_across_arms_given_shared_generator(self):
        """The dense/sparse contrast must isolate the reward: same goals+generator => same Φ."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            with open(Path(tmp) / "d.jsonl", "w"):
                pass
            _dm, _ = self._run("dense", tmp)
            dense_roll = (Path(tmp) / "dense" / "rollouts.jsonl").read_text()
            _sm, _ = self._run("sparse", tmp)
            sparse_roll = (Path(tmp) / "sparse" / "rollouts.jsonl").read_text()
        # phi_trace + success are reward-independent; extract and compare
        import json
        d = [(_j["goal"], tuple(_j["phi_trace"])) for _j in map(json.loads, dense_roll.splitlines())]
        s = [(_j["goal"], tuple(_j["phi_trace"])) for _j in map(json.loads, sparse_roll.splitlines())]
        self.assertEqual(d, s)


class SkillchainRawTraceTest(unittest.TestCase):
    """Every generated decision must survive to turns.jsonl.

    Regression guard: the rollout carries the raw text only in memory, and an earlier version of the
    trainer consumed `res["turns"]` for token accounting while writing only aggregates -- so the raw
    responses backing every Phi/ASR number were silently discarded and no run could be re-audited.
    """

    def setUp(self):
        torch.manual_seed(0)
        self.goals = trainer.build_goals(m=3, n_distractors=7, n_train=8, n_cal=4, n_ood=4, seed=0)
        self.schedule = trainer.build_schedule(n_train=8, n_goals_per_step=4, G=6, steps=2, seed=0)

    def _train(self, tmp):
        trainer.run_skillchain_training(
            model=TinyLoRALM(), generator=make_gen(0.35, seed=7), goals=self.goals["train"],
            schedule=self.schedule, arm="dense", steps=2, T=4, tau=1.0, pad_token_id=0,
            run_dir=Path(tmp), seed=0,
        )
        import json
        path = Path(tmp) / "turns.jsonl"
        self.assertTrue(path.exists(), "turns.jsonl was not written")
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]

    def test_every_turn_persists_raw_response_and_prompt(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            rows = self._train(tmp)
        self.assertTrue(rows, "no turn records written")
        for r in rows:
            self.assertIsInstance(r["response"], str)
            self.assertTrue(r["response"].strip(), "raw response persisted empty")
            self.assertTrue(r["prompt_messages"], "prompt not persisted with the response")
            for key in ("step", "arm", "goal", "turn", "phi", "advantage", "action"):
                self.assertIn(key, r)

    def test_turn_count_matches_decisions_counted_for_training(self):
        """The trace denominator must equal the training denominator -- no dropped decisions."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            model = TinyLoRALM()
            rows = trainer.run_skillchain_training(
                model=model, generator=make_gen(0.35, seed=7), goals=self.goals["train"],
                schedule=self.schedule, arm="dense", steps=2, T=4, tau=1.0, pad_token_id=0,
                run_dir=Path(tmp), seed=0,
            )
            import json
            turns = [json.loads(l) for l in
                     (Path(tmp) / "turns.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(turns), sum(r["n_total_decisions"] for r in rows))

    def test_log_turn_refuses_a_record_without_raw_response(self):
        """Fail-closed: a turn record with no raw text is an evidence hole, not a warning."""
        import tempfile
        from src.trace import TraceLogger
        with tempfile.TemporaryDirectory() as tmp:
            with TraceLogger(tmp) as t:
                with self.assertRaises(ValueError):
                    t.log_turn({"step": 1, "phi": 0.0})

    def test_reward_mode_transform(self):
        """Phase-1 placebo wiring: phantom Phi_T is informationless (independent of the trace) and
        drawn from the fixed level set; shuffled keeps Phi_T; sparse is never reshaped."""
        import random
        tr = [0.25, 0.5, 0.75, 1.0]
        # phantom Phi_T only from the fixed level set, and independent of the trace's competence
        levels = {0.0, 0.25, 0.5, 0.75, 1.0}
        phi_ts = {round(trainer._reward_trace(tr, "dense", "phantom", random.Random(i))[-1], 4)
                  for i in range(200)}
        self.assertTrue(phi_ts.issubset(levels) and len(phi_ts) >= 3)
        a = trainer._reward_trace([0.25, 0.5, 0.75, 1.0], "dense", "phantom", random.Random(7))
        b = trainer._reward_trace([0.0, 0.0, 0.0, 0.0], "dense", "phantom", random.Random(7))
        self.assertEqual(a, b)  # same RNG => identical phantom, regardless of trace competence
        sh = trainer._reward_trace(tr, "dense", "shuffled", random.Random(3))
        self.assertAlmostEqual(sh[-1], 1.0)                       # shuffled keeps Phi_T
        self.assertEqual(trainer._reward_trace(tr, "sparse", "phantom", random.Random(1)), tr)
        self.assertEqual(trainer._reward_trace(tr, "dense", "scored", random.Random(1)), tr)


if __name__ == "__main__":
    unittest.main(verbosity=1)
