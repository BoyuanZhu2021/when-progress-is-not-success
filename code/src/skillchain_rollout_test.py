"""CPU goldens for the skill-chain rollout + text protocol (no model, no GPU)."""
from __future__ import annotations

import random
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.domains.skillchain_env import build_chain, hop_transform, make_tool_pool  # noqa: E402
from src.skillchain_rollout import (  # noqa: E402
    initial_messages,
    parse_action,
    skillchain_rollout_batch,
    SkillChainEnv,
)


def make_mock_generator(recall_prob: float, seed: int):
    """A mock policy that reads ONLY the messages: it recovers the correct token for the current
    step from the issued-token hint in the history (with prob ``recall_prob``), else guesses among
    the shown candidates. This both drives the rollout and proves the protocol text is sufficient.

    It is a *pure per-item* function -- randomness is seeded from the item's own message history,
    so its output does not depend on batch order (which is what lets batched == sequential).
    """
    def gen(batch_messages):
        out = []
        for messages in batch_messages:
            obs = messages[-1]["content"]
            tool = re.search(r"Call tool `([^`]+)`", obs).group(1)
            candidates = re.search(r"Candidate tokens: (.+)", obs).group(1).split()
            # The SOURCE token for THIS step was issued (for this tool) in an earlier message; the
            # valid token is hop_transform(source). A "skilled" mock recalls it and applies the rule.
            source = None
            for m in messages:
                hit = re.search(rf"\({re.escape(tool)}\):\s*(\S+)", m["content"])
                if hit:
                    source = hit.group(1)
            correct = hop_transform(source) if source is not None else None
            # Per-item RNG: keyed on this trajectory's own transcript, so batch order is irrelevant.
            transcript = "\n".join(m["content"] for m in messages)
            rng = random.Random(f"{seed}|{transcript}")
            if correct is not None and rng.random() < recall_prob:
                token = correct
            else:
                token = rng.choice(candidates)
            out.append({"text": f"ACTION {tool} {token}"})
        return out

    return gen


class ParseActionTest(unittest.TestCase):
    def test_clean_action(self):
        self.assertEqual(parse_action("ACTION lookup key-1", ("lookup", "fetch")), ("lookup", "key-1"))

    def test_ignores_unknown_tool(self):
        self.assertEqual(parse_action("ACTION bogus key-1", ("lookup",)), (None, None))

    def test_fallback_finds_tool_and_next_token(self):
        self.assertEqual(parse_action("I will use fetch with abc now", ("fetch",)), ("fetch", "with"))

    def test_unparseable(self):
        self.assertEqual(parse_action("no idea", ("lookup",)), (None, None))


class ProtocolTest(unittest.TestCase):
    def test_initial_message_issues_source_tokens_and_lists_candidates(self):
        spec = build_chain("p1", m=3, n_distractors=3, tool_pool=make_tool_pool(8))
        msgs = initial_messages(SkillChainEnv(spec))
        task = msgs[-1]["content"]
        self.assertIn(spec.tools[0], task)
        self.assertIn(spec.raw[0], task)               # step 1's SOURCE token is issued up front
        self.assertIn(spec.raw[1], task)               # and step 2's source is issued for recall
        self.assertNotIn(spec.keys[0], [spec.raw[0]])  # the accepted token != the source
        for cand in [spec.keys[0], *spec.distractors[0]]:
            self.assertIn(cand, task)                   # accepted token + distractors are shown
        self.assertNotIn(spec.raw[0], [spec.keys[0], *spec.distractors[0]])  # source not a candidate


class RolloutTest(unittest.TestCase):
    def _specs(self, n, m=4, D=3):
        pool = make_tool_pool(8)
        return [build_chain(f"r|{i}", m=m, n_distractors=D, tool_pool=pool) for i in range(n)]

    def test_perfect_recall_completes_every_chain(self):
        specs = self._specs(6, m=4, D=5)
        results = skillchain_rollout_batch(specs, make_mock_generator(1.0, seed=1), T=6)
        for r in results:
            self.assertTrue(r["success"])
            self.assertEqual(r["phi_trace"][-1], 1.0)
            # monotone, one hop per turn under perfect recall
            self.assertEqual([round(p, 3) for p in r["phi_trace"]], [0.25, 0.5, 0.75, 1.0])

    def test_zero_recall_rarely_completes_and_phi_is_monotone(self):
        specs = self._specs(200, m=4, D=7)  # guess among 8 -> per-hop ~1/8
        results = skillchain_rollout_batch(specs, make_mock_generator(0.0, seed=2), T=6)
        succ = sum(r["success"] for r in results)
        self.assertLess(succ, 20)  # ~ (1/8)^4 * 200 ~ tiny; capability gating makes finishing rare
        for r in results:
            self.assertTrue(all(b >= a for a, b in zip(r["phi_trace"], r["phi_trace"][1:])))

    def test_recall_beats_guessing_on_success_rate(self):
        specs = self._specs(300, m=3, D=7)
        hi = skillchain_rollout_batch(specs, make_mock_generator(1.0, seed=3), T=5)
        lo = skillchain_rollout_batch(specs, make_mock_generator(0.0, seed=3), T=5)
        self.assertGreater(sum(r["success"] for r in hi), sum(r["success"] for r in lo))

    def test_turns_carry_generation_fields_for_the_update(self):
        specs = self._specs(3, m=2, D=1)

        def gen(batch_messages):
            # emit ids so the trainer would have something to weight; text drives the env
            out = []
            for messages in batch_messages:
                obs = messages[-1]["content"]
                tool = re.search(r"Call tool `([^`]+)`", obs).group(1)
                tok = re.search(r"Candidate tokens: (.+)", obs).group(1).split()[0]
                out.append({"text": f"ACTION {tool} {tok}", "prompt_ids": [1, 2], "resp_ids": [3]})
            return out

        results = skillchain_rollout_batch(specs, gen, T=3)
        for r in results:
            for turn in r["turns"]:
                self.assertIn("prompt_ids", turn)
                self.assertIn("resp_ids", turn)
                self.assertIn("phi", turn)

    def test_batched_equals_sequential(self):
        """Lockstep batching must equal looping one spec at a time (pure per-item gen)."""
        specs = self._specs(5, m=3, D=3)
        batched = skillchain_rollout_batch(specs, make_mock_generator(0.7, seed=9), T=5)
        sequential = [
            skillchain_rollout_batch([s], make_mock_generator(0.7, seed=9), T=5)[0]
            for s in specs
        ]
        # same generator seed + deterministic env -> identical phi traces regardless of batching
        self.assertEqual([r["phi_trace"] for r in batched], [r["phi_trace"] for r in sequential])


if __name__ == "__main__":
    unittest.main(verbosity=1)
