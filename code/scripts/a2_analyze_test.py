"""Goldens for a2_analyze's trajectory / decay / mechanism modes (added with EXP-2026W36-004).

Why these exist: the single-step path answered "is arm A ahead of arm B at step N", and that question
cannot distinguish a durable advantage from a decaying one. On the reference campaign `sft - sparse`
is +0.107 at step 8 and +0.046 at step 16 over the SAME 16 seeds, and the decline is monotone across
six consecutive read points -- so a one-point read of +0.046 reports the tail of a decaying curve as
if it were a level. These tests pin the three properties that claim depends on:

  * the decay is computed PER SEED (gap@lo - gap@hi for one seed, then averaged), never as a
    difference of two independently-averaged gaps, which would lose the pairing;
  * `--arms` really selects the pair, and the default stays dense,sparse so existing callers are
    bit-identical;
  * the mechanism ratio divides OOD gain by TRAIN gain per seed and drops seeds whose denominator is
    ~0, because a ratio with a vanishing denominator is noise amplified, not evidence.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from a2_analyze import analyze, mechanism, trajectory  # noqa: E402


def make_run(root: Path, arm: str, seed: int, evals: dict, train: dict | None = None) -> Path:
    """evals = {step: ood_asr}; train = {step: train_success}."""
    d = root / f"s{seed}_{arm}"
    d.mkdir(parents=True)
    (d / "run_meta.json").write_text(json.dumps({"arm": arm, "seed": seed}), encoding="utf-8")
    lines = []
    for st, ts in sorted((train or {}).items()):
        lines.append(json.dumps({"step": st, "train_success": ts}))
    for st, v in sorted(evals.items()):
        lines.append(json.dumps({"eval": True, "step": st, "ood_asr": v}))
    (d / "progress.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return d


class Base(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="a2t_"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)


class DecayTest(Base):
    def test_decay_is_paired_per_seed_not_a_difference_of_means(self):
        """Two seeds whose gaps move in OPPOSITE directions by equal amounts.

        The mean gap is identical at both steps, so an unpaired reading would report zero decay and
        zero spread. The paired reading must report zero mean but NON-zero sd -- that is the whole
        reason the contrast is computed per seed.
        """
        runs = [
            make_run(self.root, "sft", 1, {8: 0.60, 16: 0.50}),
            make_run(self.root, "sparse", 1, {8: 0.50, 16: 0.50}),   # gap +0.10 -> 0.00
            make_run(self.root, "sft", 2, {8: 0.50, 16: 0.60}),
            make_run(self.root, "sparse", 2, {8: 0.50, 16: 0.50}),   # gap  0.00 -> +0.10
        ]
        r = trajectory(runs, "ood_asr", [8, 16], ("sft", "sparse"), None, "t")
        self.assertAlmostEqual(r["per_step"][8]["paired_t"]["effect"], 0.05, places=6)
        self.assertAlmostEqual(r["per_step"][16]["paired_t"]["effect"], 0.05, places=6)
        self.assertAlmostEqual(r["decay"]["paired_t"]["effect"], 0.0, places=6)
        self.assertGreater(r["decay"]["paired_t"]["sd_between"], 0.10)
        self.assertEqual(r["decay"]["paired_t"]["per_seed_diffs"], [0.1, -0.1])

    def test_decay_sign_is_gap_at_first_minus_gap_at_last(self):
        """A SHRINKING advantage must come out POSITIVE, matching how the EXP reports it."""
        runs = [
            make_run(self.root, "sft", 1, {8: 0.60, 16: 0.60}),
            make_run(self.root, "sparse", 1, {8: 0.40, 16: 0.55}),   # gap +0.20 -> +0.05
        ]
        r = trajectory(runs, "ood_asr", [8, 16], ("sft", "sparse"), None, "t")
        self.assertAlmostEqual(r["decay"]["paired_t"]["effect"], 0.15, places=6)
        self.assertEqual(r["decay"]["from_step"], 8)
        self.assertEqual(r["decay"]["to_step"], 16)

    def test_decay_uses_first_and_last_available_step_not_the_requested_extremes(self):
        """A step nobody evaluated must be skipped, and the decay anchored on what actually exists --
        otherwise a typo'd --steps silently yields an empty or half-populated contrast."""
        runs = [
            make_run(self.root, "sft", 1, {8: 0.60, 16: 0.50}),
            make_run(self.root, "sparse", 1, {8: 0.40, 16: 0.40}),
        ]
        r = trajectory(runs, "ood_asr", [4, 8, 16, 99], ("sft", "sparse"), None, "t")
        self.assertEqual(r["steps"], [8, 16])
        self.assertEqual((r["decay"]["from_step"], r["decay"]["to_step"]), (8, 16))

    def test_per_arm_delta_attributes_the_decay_to_an_arm(self):
        """The decay can come from the treatment stalling or the control catching up; the EXP's claim
        is specifically that SPARSE rises. That is only checkable if each arm's own change is reported."""
        runs = [
            make_run(self.root, "sft", 1, {8: 0.60, 16: 0.62}),
            make_run(self.root, "sparse", 1, {8: 0.40, 16: 0.58}),
        ]
        r = trajectory(runs, "ood_asr", [8, 16], ("sft", "sparse"), None, "t")
        self.assertAlmostEqual(r["per_arm_delta"]["sft"]["effect"], 0.02, places=6)
        self.assertAlmostEqual(r["per_arm_delta"]["sparse"]["effect"], 0.18, places=6)

    def test_only_seeds_present_in_both_arms_are_paired(self):
        runs = [
            make_run(self.root, "sft", 1, {8: 0.60, 16: 0.60}),
            make_run(self.root, "sparse", 1, {8: 0.40, 16: 0.40}),
            make_run(self.root, "sft", 7, {8: 0.90, 16: 0.90}),   # no sparse partner
        ]
        r = trajectory(runs, "ood_asr", [8, 16], ("sft", "sparse"), None, "t")
        self.assertEqual(r["per_step"][8]["n"], 1)

    def test_seeds_filter_restricts_the_pairing(self):
        runs = [
            make_run(self.root, "sft", 1, {8: 0.60, 16: 0.60}),
            make_run(self.root, "sparse", 1, {8: 0.40, 16: 0.40}),
            make_run(self.root, "sft", 2, {8: 0.10, 16: 0.10}),
            make_run(self.root, "sparse", 2, {8: 0.90, 16: 0.90}),
        ]
        r = trajectory(runs, "ood_asr", [8, 16], ("sft", "sparse"), {1}, "t")
        self.assertEqual(r["per_step"][8]["n"], 1)
        self.assertAlmostEqual(r["per_step"][8]["paired_t"]["effect"], 0.20, places=6)


class ArmSelectionTest(Base):
    def test_arms_argument_selects_the_pair(self):
        runs = [
            make_run(self.root, "sft", 1, {8: 0.60}),
            make_run(self.root, "sparse", 1, {8: 0.40}),
            make_run(self.root, "dense", 1, {8: 0.10}),
        ]
        sft = trajectory(runs, "ood_asr", [8], ("sft", "sparse"), None, "t")
        dense = trajectory(runs, "ood_asr", [8], ("dense", "sparse"), None, "t")
        self.assertAlmostEqual(sft["per_step"][8]["paired_t"]["effect"], +0.20, places=6)
        self.assertAlmostEqual(dense["per_step"][8]["paired_t"]["effect"], -0.30, places=6)

    def test_single_step_path_is_unchanged_and_still_dense_vs_sparse(self):
        """Regression: `analyze` is what every pre-EXP-2026W36-004 caller uses. It must keep reading
        the dense/sparse pair from the same fields and returning the same keys."""
        runs = [
            make_run(self.root, "dense", 1, {12: 0.30}),
            make_run(self.root, "sparse", 1, {12: 0.40}),
            make_run(self.root, "dense", 2, {12: 0.35}),
            make_run(self.root, "sparse", 2, {12: 0.40}),
        ]
        r = analyze(runs, "ood_asr", 12, None, "regression")
        self.assertEqual(r["n"], 2)
        self.assertAlmostEqual(r["dense_mean"], 0.325, places=6)
        self.assertAlmostEqual(r["sparse_mean"], 0.40, places=6)
        self.assertAlmostEqual(r["paired_t"]["effect"], -0.075, places=6)
        self.assertFalse(r["SUPPORT_rule_met"])


class MechanismTest(Base):
    """The claim is 'sft keeps LEARNING but stops GENERALISING', i.e. its OOD gain per unit of
    train_success gain falls below sparse's. These pin how that ratio is formed."""

    def _pair(self, sft_ood, sft_tr, sp_ood, sp_tr, seed=1):
        return [
            make_run(self.root, "sft", seed, sft_ood, sft_tr),
            make_run(self.root, "sparse", seed, sp_ood, sp_tr),
        ]

    def test_ratio_is_ood_gain_over_train_gain(self):
        runs = self._pair(
            {8: 0.40, 16: 0.50}, {s: (0.20 if s <= 8 else 0.40) for s in range(1, 17)},
            {8: 0.40, 16: 0.60}, {s: (0.20 if s <= 8 else 0.40) for s in range(1, 17)})
        m = mechanism(runs, "ood_asr", [8, 16], ("sft", "sparse"), None, "t")
        self.assertAlmostEqual(m["sft_ratio_mean"], 0.5, places=4)     # 0.10 OOD / 0.20 train
        self.assertAlmostEqual(m["sparse_ratio_mean"], 1.0, places=4)  # 0.20 OOD / 0.20 train

    def test_seed_with_no_training_gain_is_dropped_not_divided_by_zero(self):
        """A flat-training seed makes the denominator ~0; including it would turn eval noise into an
        arbitrarily large ratio and dominate the mean."""
        runs = self._pair(
            {8: 0.40, 16: 0.50}, {s: 0.30 for s in range(1, 17)},   # train perfectly flat
            {8: 0.40, 16: 0.60}, {s: 0.30 for s in range(1, 17)})
        m = mechanism(runs, "ood_asr", [8, 16], ("sft", "sparse"), None, "t")
        self.assertEqual(m["n"], 0)

    def test_train_halves_do_not_overlap_and_cover_the_budget(self):
        """Steps 1-8 vs 9-16 for eval points [2,4,6,8] vs [10,12,14,16]: an off-by-one that let step 8
        into both halves would make the two arms' denominators share a term and shrink the contrast."""
        tr = {s: 0.10 * s for s in range(1, 17)}
        runs = self._pair({2: .1, 4: .2, 6: .3, 8: .4, 10: .5, 12: .6, 14: .7, 16: .8}, tr,
                          {2: .1, 4: .2, 6: .3, 8: .4, 10: .5, 12: .6, 14: .7, 16: .8}, tr)
        m = mechanism(runs, "ood_asr", [2, 4, 6, 8, 10, 12, 14, 16], ("sft", "sparse"), None, "t")
        self.assertEqual(m["first_half"], [2, 4, 6, 8])
        self.assertEqual(m["second_half"], [10, 12, 14, 16])
        # identical arms -> identical ratios -> a zero paired contrast, not a spurious one
        self.assertAlmostEqual(m["sft_ratio_mean"], m["sparse_ratio_mean"], places=9)

    def test_paired_contrast_is_reported_when_more_than_one_seed(self):
        runs = []
        for seed, (a, b) in enumerate([(0.50, 0.60), (0.52, 0.63)], start=1):
            tr = {s: (0.20 if s <= 8 else 0.40) for s in range(1, 17)}
            runs += [make_run(self.root, "sft", seed, {8: 0.40, 16: a}, tr),
                     make_run(self.root, "sparse", seed, {8: 0.40, 16: b}, tr)]
        m = mechanism(runs, "ood_asr", [8, 16], ("sft", "sparse"), None, "t")
        self.assertEqual(m["n"], 2)
        self.assertLess(m["paired_t"]["effect"], 0.0)   # sft generalises worse
        self.assertEqual(m["paired_t"]["sign_dense_gt_sparse"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
