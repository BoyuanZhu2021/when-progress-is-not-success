"""CPU goldens for the deterministic skill-chain environment."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.domains.skillchain_env import (  # noqa: E402
    SkillChainEnv,
    SkillChainSpec,
    build_chain,
    make_tool_pool,
    phi_of_hop,
)


class GateTest(unittest.TestCase):
    def _spec(self, m=4, D=3):
        return build_chain(f"g|{m}|{D}", m=m, n_distractors=D, tool_pool=make_tool_pool(8))

    def test_correct_key_advances_one_hop(self):
        env = SkillChainEnv(self._spec())
        obs = env.reset()
        self.assertEqual(env.phi, 0.0)
        _obs, phi, done, info = env.step(obs["tool"], env.spec.keys[0])
        self.assertTrue(info["correct"])
        self.assertEqual(env.hop, 1)
        self.assertAlmostEqual(phi, 0.25)
        self.assertFalse(done)

    def test_wrong_key_wastes_a_call_no_progress(self):
        env = SkillChainEnv(self._spec())
        obs = env.reset()
        wrong = env.spec.distractors[0][0]
        _obs, phi, done, info = env.step(obs["tool"], wrong)
        self.assertFalse(info["correct"])
        self.assertEqual(env.hop, 0)
        self.assertEqual(phi, 0.0)
        self.assertEqual(env.n_calls, 1)  # the wasted attempt is still counted

    def test_cannot_skip_ahead_hop2_key_fails_at_hop1(self):
        """Capability gating: hop 2's key does not open hop 1."""
        env = SkillChainEnv(self._spec())
        obs = env.reset()
        _obs, _phi, _done, info = env.step(obs["tool"], env.spec.keys[1])  # key for the NEXT hop
        self.assertFalse(info["correct"])
        self.assertEqual(env.hop, 0)

    def test_wrong_tool_right_key_fails(self):
        env = SkillChainEnv(self._spec())
        obs = env.reset()
        other_tool = next(t for t in env.spec.tools if t != obs["tool"])
        _obs, _phi, _done, info = env.step(other_tool, env.spec.keys[0])
        self.assertFalse(info["correct"])

    def test_full_chain_completes_and_phi_is_monotone(self):
        env = SkillChainEnv(self._spec(m=4))
        env.reset()
        trace = [env.phi]
        for k in range(env.spec.m):
            _obs, phi, done, _info = env.step(env.spec.tools[k], env.spec.keys[k])
            trace.append(phi)
        self.assertTrue(env.success)
        self.assertTrue(done)
        self.assertEqual(trace, [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertTrue(all(b >= a for a, b in zip(trace, trace[1:])))  # monotone

    def test_candidates_contain_exactly_one_true_key(self):
        env = SkillChainEnv(self._spec(D=5))
        for hop in range(env.spec.m):
            cands = env.candidates(hop)
            self.assertEqual(len(cands), 6)  # 1 true + 5 distractors
            self.assertEqual(sum(c == env.spec.keys[hop] for c in cands), 1)

    def test_observation_and_candidate_order_are_deterministic(self):
        spec = self._spec(D=5)
        a = SkillChainEnv(spec).candidates(1)
        b = SkillChainEnv(spec).candidates(1)
        self.assertEqual(a, b)  # reproducible across env instances (CRN across arms)

    def test_D0_has_single_candidate_so_hop_is_trivial(self):
        env = SkillChainEnv(self._spec(D=0))
        self.assertEqual(env.candidates(0), [env.spec.keys[0]])


class SpecTest(unittest.TestCase):
    def test_true_key_never_among_distractors(self):
        for D in (0, 1, 4):
            spec = build_chain(f"x|{D}", m=3, n_distractors=D, tool_pool=make_tool_pool(6))
            for k in range(spec.m):
                self.assertNotIn(spec.keys[k], spec.distractors[k])

    def test_tools_within_a_chain_are_distinct(self):
        spec = build_chain("y", m=5, n_distractors=2, tool_pool=make_tool_pool(8))
        self.assertEqual(len(set(spec.tools)), 5)

    def test_build_is_deterministic_from_seed(self):
        a = build_chain("same-seed", m=4, n_distractors=3, tool_pool=make_tool_pool(8))
        b = build_chain("same-seed", m=4, n_distractors=3, tool_pool=make_tool_pool(8))
        self.assertEqual(a, b)

    def test_different_seeds_differ(self):
        a = build_chain("seed-a", m=4, n_distractors=3, tool_pool=make_tool_pool(8))
        b = build_chain("seed-b", m=4, n_distractors=3, tool_pool=make_tool_pool(8))
        self.assertNotEqual(a.keys, b.keys)

    def test_phi_of_hop_bounds(self):
        self.assertEqual(phi_of_hop(0, 4), 0.0)
        self.assertEqual(phi_of_hop(4, 4), 1.0)
        with self.assertRaises(ValueError):
            phi_of_hop(5, 4)

    def test_m_beyond_tool_pool_rejected(self):
        with self.assertRaises(ValueError):
            build_chain("z", m=9, n_distractors=1, tool_pool=make_tool_pool(8))


if __name__ == "__main__":
    unittest.main(verbosity=1)
