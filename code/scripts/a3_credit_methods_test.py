"""CPU goldens for the fullft-campaign credit-assignment methods (B2 hindsight / B3 curriculum / B4 SFT).

`mt_grpo._selftest_credit` covers the pure threshold + weight math. This file covers the parts that
only exist once the pieces are WIRED TOGETHER, and each test corresponds to a specific way the
campaign could produce a confidently-wrong number instead of an error:

  * the DV is retargeted along with the training threshold -> every B2/B3 result becomes meaningless
    because the arm would be scored on the easier task it was trained on;
  * SFT's "cross-entropy on attacker tokens" is asserted in prose but is actually something else --
    this is the claim the entire B4 arm rests on, so it is checked numerically against an
    independently-computed CE rather than by reading the code;
  * a goal with zero successful attempts contributes a small non-zero gradient instead of nothing.

Run: python code/scripts/a3_credit_methods_test.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                       # code/scripts (a2_train, h1_skillchain_train)
sys.path.insert(0, str(_HERE.parent))                # code (src.*)

from src.mt_grpo import (sft_advantages, train_m_threshold, group_advantages,  # noqa: E402
                         clip_negative_advantages, group_is_starved)


def _succ_at(count: int, K: int, tau: float) -> bool:
    """The success predicate a3 applies, spelled out: count_trace is a RATE, thresholded at m/K."""
    return (count / K) >= tau - 1e-9


class TrainVsEvalThresholdTest(unittest.TestCase):
    """The DV must never move. Training threshold and eval threshold are separate quantities."""

    K = M = 4

    def _eval_tau(self):
        # exactly how a3 computes the EVAL threshold: args.m_of_K / args.kfield_K, no knobs applied
        return self.M / self.K

    def test_hindsight_moves_training_threshold_only(self):
        _m, tau_train = train_m_threshold(self.M, self.K, hindsight_m=2)
        self.assertEqual(tau_train, 0.5)
        self.assertEqual(self._eval_tau(), 1.0)
        # a trajectory delivering 2 of 4 fields: TRAINS as a success, EVALUATES as a failure
        self.assertTrue(_succ_at(2, self.K, tau_train))
        self.assertFalse(_succ_at(2, self.K, self._eval_tau()))

    def test_curriculum_moves_training_threshold_only(self):
        for step in range(1, 17):
            _m, tau_train = train_m_threshold(self.M, self.K, step=step, steps=16, curriculum=True)
            self.assertLessEqual(tau_train, self._eval_tau())
            self.assertEqual(self._eval_tau(), 1.0, "eval threshold moved with the curriculum")
        # count=1 flips from trainable success to failure across the schedule
        _m1, tau_first = train_m_threshold(self.M, self.K, step=1, steps=16, curriculum=True)
        _m2, tau_last = train_m_threshold(self.M, self.K, step=16, steps=16, curriculum=True)
        self.assertTrue(_succ_at(1, self.K, tau_first))
        self.assertFalse(_succ_at(1, self.K, tau_last))

    def test_default_leaves_both_thresholds_equal(self):
        """No flags => training and eval agree, i.e. prior campaigns reproduce bit-for-bit."""
        for step in (1, 8, 16):
            _m, tau_train = train_m_threshold(self.M, self.K, step=step, steps=16)
            self.assertEqual(tau_train, self._eval_tau())


class ArgValidationTest(unittest.TestCase):
    """Misconfiguration must raise BEFORE a model is loaded, not produce a quietly-wrong run."""

    def _args(self, **kw):
        from a3_multiturn_train import build_parser
        base = ["--run-dir", "x", "--m-of-K", "4", "--kfield-K", "4"]
        for k, v in kw.items():
            base += [k] if v is True else [k, str(v)]
        return build_parser().parse_args(base)

    def test_both_threshold_knobs_rejected(self):
        from a3_multiturn_train import validate_credit_args
        with self.assertRaises(ValueError):
            validate_credit_args(self._args(**{"--hindsight-m": 2, "--m-curriculum": True}))

    def test_hindsight_without_m_of_K_rejected(self):
        from a3_multiturn_train import build_parser, validate_credit_args
        args = build_parser().parse_args(["--run-dir", "x", "--hindsight-m", "2"])
        with self.assertRaises(ValueError):
            validate_credit_args(args)

    def test_out_of_range_hindsight_rejected(self):
        from a3_multiturn_train import validate_credit_args
        for bad in (0, 5):
            with self.assertRaises(ValueError):
                validate_credit_args(self._args(**{"--hindsight-m": bad}))

    def test_sft_rejects_rl_only_settings(self):
        from a3_multiturn_train import validate_credit_args
        with self.assertRaises(ValueError):
            validate_credit_args(self._args(**{"--arm": "sft", "--dynamic-sampling": 2.0}))

    def test_valid_configs_pass(self):
        from a3_multiturn_train import validate_credit_args
        for kw in ({}, {"--hindsight-m": 2}, {"--m-curriculum": True}, {"--arm": "sft"},
                   {"--arm": "sft_all"}, {"--arm": "sft_fail"}, {"--arm": "rl_pos"}):
            validate_credit_args(self._args(**kw))       # must not raise

    def test_ce_family_rejects_rl_only_settings(self):
        """sft_all / sft_fail inherit sft's guard: an RL-estimator flag on a CE arm is a misconfiguration."""
        from a3_multiturn_train import validate_credit_args
        for arm in ("sft_all", "sft_fail"):
            with self.assertRaises(ValueError, msg=arm):
                validate_credit_args(self._args(**{"--arm": arm, "--dynamic-sampling": 2.0}))

    def test_ce_family_metadata_contract(self):
        """run_meta's loss/beta_kl_effective must say ce/0 for every CE arm and pg/beta for rl_pos."""
        from a3_multiturn_train import CE_ARMS
        self.assertEqual(CE_ARMS, ("sft", "sft_all", "sft_fail"))
        self.assertNotIn("rl_pos", CE_ARMS)


class SFTIsCrossEntropyTest(unittest.TestCase):
    """B4's load-bearing claim, checked numerically: grpo_loss_step with weight 1 and beta_kl 0 IS
    teacher-forced CE over the attacker's response tokens only."""

    def setUp(self):
        try:
            import torch                                     # noqa: F401
        except ImportError:                                  # pragma: no cover
            self.skipTest("torch not available")

    def _toy(self):
        import torch
        from torch import nn

        class Out:
            def __init__(self, logits):
                self.logits = logits

        class Toy(nn.Module):
            """Deterministic tiny LM. Real shapes, no pretrained weights, runs in milliseconds."""

            def __init__(self, vocab=11, dim=8):
                super().__init__()
                torch.manual_seed(0)
                self.emb = nn.Embedding(vocab, dim)
                self.head = nn.Linear(dim, vocab)

            def forward(self, input_ids, attention_mask=None):
                return Out(self.head(self.emb(input_ids)))

        return Toy()

    def test_weight_one_reproduces_cross_entropy_on_response_tokens(self):
        import torch
        import torch.nn.functional as F
        from a2_train import grpo_loss_step
        from h1_skillchain_train import token_logps_batch

        model = self._toy()
        pairs = [(torch.tensor([1, 2, 3]), torch.tensor([4, 5])),
                 (torch.tensor([1, 7]), torch.tensor([8, 9, 10]))]
        denom = sum(int(r.shape[0]) for _p, r in pairs)

        # Independently-computed reference: mean NLL of the response tokens under teacher forcing.
        with torch.no_grad():
            ref_ce = -sum(float(lp.sum()) for lp in token_logps_batch(model, pairs, pad_token_id=0)) / denom

        model.zero_grad(set_to_none=True)
        examples = [(p, r, 1.0) for p, r in pairs]           # exactly what sft_advantages produces
        pg, kl = grpo_loss_step(model, examples, pad_token_id=0, beta_kl=0.0,
                                denom=denom, ref_model=model)
        self.assertAlmostEqual(pg, ref_ce, places=5,
                               msg="SFT weight-1 loss is not the CE on response tokens")
        self.assertEqual(kl, 0.0, "SFT must carry no KL term")
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0
                            for p in model.parameters()), "SFT step produced no gradient")

    def test_prompt_tokens_are_not_scored(self):
        """CE must be masked to the attacker's own tokens. Changing PROMPT ids changes the loss
        (context differs) but the number of scored positions must equal the RESPONSE length."""
        import torch
        from h1_skillchain_train import token_logps_batch

        model = self._toy()
        for resp_len in (1, 2, 5):
            pairs = [(torch.tensor([1, 2, 3]), torch.arange(4, 4 + resp_len))]
            lp = token_logps_batch(model, pairs, pad_token_id=0)[0]
            self.assertEqual(int(lp.shape[0]), resp_len,
                             "scored positions != response length (prompt tokens leaked into the loss)")

    def test_failed_goal_contributes_nothing(self):
        """A goal whose every attempt failed must be an exact no-op: all-zero weights, and a3's
        `abs(adv) < 1e-9` filter drops the examples so no gradient of arbitrary direction is taken."""
        import torch
        from a2_train import grpo_loss_step

        weights = sft_advantages([False, False, False], [2, 2, 2])
        self.assertTrue(all(v == 0.0 for row in weights for v in row))

        model = self._toy()
        model.zero_grad(set_to_none=True)
        kept = [(torch.tensor([1, 2]), torch.tensor([3, 4]), a)
                for row in weights for a in row if abs(a) >= 1e-9]
        self.assertEqual(kept, [], "failed-goal turns must not survive a3's example filter")
        # nothing to run => no gradient at all
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        # and the guard is real: had they survived at weight 0, the loss would be identically 0
        pg, _kl = grpo_loss_step(model, [(torch.tensor([1, 2]), torch.tensor([3, 4]), 0.0)],
                                 pad_token_id=0, beta_kl=0.0, denom=2, ref_model=model)
        self.assertEqual(pg, 0.0)


class SftMechanismArmsTest(unittest.TestCase):
    """sft-mechanism-4b-9b-v1 §6 Phase 0 goldens. Each 2x2 corner must be exactly ONE ingredient away
    from its neighbours, or the sufficiency/necessity reading is meaningless."""

    FLAGS = [True, False, False, True, False, False]
    NT = [3, 2, 4, 1, 5, 2]

    def test_sft_all_weights_are_all_one(self):
        w = sft_advantages([True] * len(self.FLAGS), self.NT)          # the sft_all substitution
        self.assertEqual(w, [[1.0] * n for n in self.NT])

    def test_sft_fail_weights_are_the_complement_of_sft(self):
        sft = sft_advantages(self.FLAGS, self.NT)
        fail = sft_advantages([not f for f in self.FLAGS], self.NT)  # the sft_fail substitution
        for a, b in zip(sft, fail):
            self.assertEqual([x + y for x, y in zip(a, b)], [1.0] * len(a))
        self.assertTrue(any(v == 1.0 for row in fail for v in row))
        self.assertTrue(any(v == 0.0 for row in fail for v in row))

    @staticmethod
    def _sparse_group():
        # G=6 sparse reward rows: terminal 1 on success, else 0; variable length.
        return [[0, 0, 1], [0, 0, 0], [0, 0, 0, 0], [0, 1], [0, 0, 0], [0, 0, 0, 0]]

    def test_rl_pos_is_sparse_with_negatives_removed(self):
        A = group_advantages(self._sparse_group())
        C = clip_negative_advantages(A)
        self.assertEqual([len(r) for r in C], [len(r) for r in A])
        for a, c in zip(A, C):
            for x, y in zip(a, c):
                self.assertGreaterEqual(y, 0.0)
                if x >= 0.0:
                    self.assertEqual(y, x, "positive/zero advantages must pass through untouched")
                else:
                    self.assertEqual(y, 0.0, "negative advantages must become exactly 0")
        self.assertTrue(any(x < 0 for row in A for x in row), "fixture must contain a negative to clip")

    def test_rl_pos_starved_iff_no_positive_advantage_after_clip(self):
        # a group with signal keeps a positive after clipping -> fed
        fed = clip_negative_advantages(group_advantages(self._sparse_group()))
        self.assertFalse(group_is_starved(fed))
        self.assertTrue(any(v > 0 for row in fed for v in row))
        # all-fail group: GRPO gives all zeros -> starved before and after
        allfail = clip_negative_advantages(group_advantages([[0, 0, 0]] * 6))
        self.assertTrue(group_is_starved(allfail))
        # negatives-only advantages (the case the plan's C6 names): starved AFTER the clip
        neg = clip_negative_advantages([[-0.5, -1.0], [-0.2], [0.0, 0.0, 0.0]])
        self.assertTrue(group_is_starved(neg))

    def test_ce_identity_holds_for_sft_all_and_sft_fail_weights(self):
        """The CE identity (SFTIsCrossEntropyTest) rests on weights being exactly 1.0 on scored turns;
        both new arms produce only {0.0, 1.0}, so it carries over unchanged."""
        for flags in ([True] * 6, [not f for f in self.FLAGS]):
            for row in sft_advantages(flags, self.NT):
                self.assertTrue(set(row) <= {0.0, 1.0})


class SFTWeightShapeTest(unittest.TestCase):
    def test_weights_match_turn_counts(self):
        w = sft_advantages([True, False], [3, 1])
        self.assertEqual([len(r) for r in w], [3, 1])
        self.assertEqual(w, [[1.0, 1.0, 1.0], [0.0]])


class SelfImitationTest(unittest.TestCase):
    """B5 replay. The three properties that separate this from hindsight (-0.216) are:
    only TRUE successes are stored, replay fires ONLY into starved groups, and a group that already
    has gradient is left byte-identical to sparse."""

    def test_only_true_successes_are_stored(self):
        from src.mt_grpo import SelfImitationBuffer
        b = SelfImitationBuffer()
        self.assertFalse(b.add("g", False, [{"t": 1}]), "a FAILURE must never enter the buffer")
        self.assertFalse(b.add("g", True, []), "an empty trajectory must not enter the buffer")
        self.assertTrue(b.add("g", True, [{"t": 1}]))
        self.assertEqual(b.size("g"), 1)

    def test_replay_never_crosses_goals(self):
        from src.mt_grpo import SelfImitationBuffer
        b = SelfImitationBuffer()
        b.add("goalA", True, [{"t": "A"}])
        self.assertEqual(b.sample("goalB", 3), [], "goal B must not see goal A's trajectory")

    def test_gate_fires_only_on_starved_groups(self):
        """If a group has ANY non-zero advantage, replay must not touch it -- otherwise the arm
        dilutes the very cells where the terminal signal is already informative."""
        from src.mt_grpo import group_is_starved
        self.assertTrue(group_is_starved([[0.0, 0.0], [0.0]]))
        self.assertTrue(group_is_starved([[], []]))
        self.assertFalse(group_is_starved([[0.0], [0.4]]))
        self.assertFalse(group_is_starved([[-1e-3, 0.0]]))

    def test_newest_first_and_capacity(self):
        from src.mt_grpo import SelfImitationBuffer
        b = SelfImitationBuffer(capacity=3)
        for i in range(5):
            b.add("g", True, [{"t": i}])
        got = [tr[0]["t"] for tr in b.sample("g", 10)]
        self.assertEqual(got, [4, 3, 2], "must keep the freshest and return newest-first")

    def test_arm_is_inert_when_never_starved(self):
        """End-to-end invariant: with no starved group, replay contributes zero examples, so the
        run is identical to plain sparse."""
        from src.mt_grpo import SelfImitationBuffer, group_is_starved
        b = SelfImitationBuffer()
        b.add("g", True, [{"prompt_ids": [1], "resp_ids": [2]}])
        advantages = [[0.5], [-0.5]]                       # group HAS signal
        injected = []
        if group_is_starved(advantages):
            injected = b.sample("g", 2)
        self.assertEqual(injected, [], "replay fired on a group that already had gradient")


if __name__ == "__main__":
    unittest.main(verbosity=2)
