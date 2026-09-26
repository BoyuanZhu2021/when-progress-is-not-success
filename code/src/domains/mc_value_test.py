"""CPU goldens for the Monte-Carlo value potential (DISC-2026W33-006).

Uses a scripted victim so the true continuation-success rate is known in closed form, which is the only
way to check that V_hat estimates what it claims to. Also locks the two properties the estimate rests on:
snapshots must RESTORE exactly, and enabling snapshots must not perturb the default rollout.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agentdojo.functions_runtime import FunctionsRuntime  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402

from src.domains.agentdojo_infogated import (  # noqa: E402
    ATTACKER_IBAN, _FAMILY_TARGETS, register_infogated_family,
)
from src.domains.agentdojo_multiturn import MTGoal, multiturn_rollout_batch  # noqa: E402
from src.domains.mc_value import mc_value_cost_multiplier, mc_value_rollout_batch  # noqa: E402

_SUITE = get_suite("v1", "banking")
_COMP = {9910: ["iban", "balance", "street", "city"]}
register_infogated_family(_SUITE, _COMP)


def _gen(batch):
    """Attacker stub: content is irrelevant to these goldens (the victim script drives everything)."""
    return [{"text": "next", "prompt_ids": None, "resp_ids": None} for _ in batch]


def _victim_after(n_ok):
    """Scripted victim: performs the FULL ground-truth chain on the n_ok-th call of each episode branch.

    Because the counter lives on the returned closure and is keyed by the message length, every forked
    continuation sees the same deterministic schedule -- so the true continuation success rate is exactly
    computable and V_hat can be checked against it rather than against itself.
    """
    pre = _SUITE.load_and_inject_default_environment({})
    gt = _SUITE.get_injection_task_by_id("injection_task_9910").ground_truth(pre)

    def step(work_env, runtime, messages, atext):
        msgs = list(messages or [])
        msgs.append({"role": "user", "content": atext})
        if len(msgs) >= n_ok:                       # from this turn on, the victim completes the chain
            for fc in gt:
                try:
                    runtime.run_function(work_env, fc.function, dict(fc.args), raise_on_error=False)
                except Exception:  # noqa: BLE001
                    pass
        msgs.append({"role": "assistant", "content": "ok", "tool_calls": []})
        return work_env, msgs
    return step


class SnapshotTest(unittest.TestCase):
    def setUp(self):
        self.goal = MTGoal("injection_task_9910", _COMP[9910])

    def test_snapshot_does_not_change_the_rollout(self):
        """Enabling snapshots must be observationally inert -- otherwise V_hat is measured on a
        different process than the one being trained."""
        a = multiturn_rollout_batch([self.goal], _gen, T=4, suite=_SUITE,
                                    victim_step_fn=_victim_after(99))[0]
        b = multiturn_rollout_batch([self.goal], _gen, T=4, suite=_SUITE, snapshot=True,
                                    victim_step_fn=_victim_after(99))[0]
        self.assertEqual(a["phi_trace"], b["phi_trace"])
        self.assertEqual(a["n_turns"], b["n_turns"])
        self.assertEqual(a["success"], b["success"])

    def test_snapshots_are_one_per_turn_and_carry_state(self):
        r = multiturn_rollout_batch([self.goal], _gen, T=4, suite=_SUITE, snapshot=True,
                                    victim_step_fn=_victim_after(99))[0]
        self.assertEqual(len(r["snapshots"]), r["n_turns"])
        self.assertEqual([s["turn"] for s in r["snapshots"]], list(range(1, r["n_turns"] + 1)))
        for s in r["snapshots"]:
            self.assertIsNotNone(s["work_env"])

    def test_resume_from_snapshot_reproduces_the_tail(self):
        """A continuation started from the snapshot at turn t must reach the same outcome the original
        episode did, when the victim schedule is deterministic. This is what makes V_hat meaningful."""
        full = multiturn_rollout_batch([self.goal], _gen, T=4, suite=_SUITE, snapshot=True,
                                       victim_step_fn=_victim_after(3))[0]
        snap = full["snapshots"][0]                                    # state after turn 1
        cont = multiturn_rollout_batch([self.goal], _gen, T=4, suite=_SUITE,
                                       init_states=[{**snap, "start_turn": snap["turn"] + 1}],
                                       victim_step_fn=_victim_after(3))[0]
        self.assertEqual(cont["success"], full["success"])


class MCValueTest(unittest.TestCase):
    def setUp(self):
        self.goal = MTGoal("injection_task_9910", _COMP[9910])

    def test_all_continuations_succeed_gives_value_one(self):
        """Victim always completes the chain => every continuation succeeds => V_hat == 1 everywhere."""
        r = mc_value_rollout_batch([self.goal], _gen, T=4, K=2, suite=_SUITE,
                                   victim_step_fn=_victim_after(1))[0]
        self.assertTrue(r["mcv_trace"], "mcv_trace must be populated")
        for v in r["mcv_trace"]:
            self.assertAlmostEqual(v, 1.0, places=9)

    def test_no_continuation_succeeds_gives_value_zero(self):
        """Victim never completes the chain => V_hat == 0 everywhere. This is the case the module's
        docstring predicts will dominate a rare-success task, collapsing dense toward sparse."""
        r = mc_value_rollout_batch([self.goal], _gen, T=4, K=2, suite=_SUITE,
                                   victim_step_fn=_victim_after(999))[0]
        self.assertFalse(r["success"])
        for v in r["mcv_trace"]:
            self.assertAlmostEqual(v, 0.0, places=9)

    def test_trace_length_matches_turns_and_K_recorded(self):
        r = mc_value_rollout_batch([self.goal], _gen, T=4, K=3, suite=_SUITE,
                                   victim_step_fn=_victim_after(999))[0]
        self.assertEqual(len(r["mcv_trace"]), r["n_turns"])
        self.assertEqual(r["mcv_K"], 3)
        self.assertNotIn("snapshots", r, "deep env copies must be dropped before logging")

    def test_dense_reward_on_mcv_telescopes_to_final_value(self):
        """r_t = Delta V_hat must sum to V_hat_T -- the same telescoping property the programmatic
        potential has, so `dense` semantics are unchanged and only the POTENTIAL differs."""
        from src.mt_grpo import per_turn_rewards
        r = mc_value_rollout_batch([self.goal], _gen, T=4, K=2, suite=_SUITE,
                                   victim_step_fn=_victim_after(1))[0]
        rew = per_turn_rewards(r["mcv_trace"], 1.0, "dense")
        self.assertAlmostEqual(sum(rew), r["mcv_trace"][-1], places=9)

    def test_K_must_be_positive(self):
        with self.assertRaises(ValueError):
            mc_value_rollout_batch([self.goal], _gen, T=4, K=0, suite=_SUITE,
                                   victim_step_fn=_victim_after(1))

    def test_cost_multiplier_matches_the_quoted_formula(self):
        self.assertAlmostEqual(mc_value_cost_multiplier(5, 2), 1.0 + (2 * 5 * 4 / 2) / 5, places=9)
        self.assertAlmostEqual(mc_value_cost_multiplier(5, 2), 5.0, places=9)   # the 5x quoted in the plan
        self.assertAlmostEqual(mc_value_cost_multiplier(1, 4), 1.0, places=9)   # T=1: nothing to fork


if __name__ == "__main__":
    unittest.main(verbosity=2)
