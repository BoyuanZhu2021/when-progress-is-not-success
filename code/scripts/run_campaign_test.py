"""Goldens for run_campaign's job planning and lane/URL plumbing.

The driver had no tests before 2026-09-03. Two things it now does are easy to get silently wrong and
would each spend a day of GPU on the wrong thing: the per-arm seed override (plan sft-mechanism-4b-9b-v1
runs sft_fail at 2-7 while the 2x2 arms run 2-11) and the per-card victim URL (each lane must talk to
its OWN co-located victim, never a neighbour's).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_campaign import build_jobs, parse_arm_seeds, parse_lanes, parse_seeds  # noqa: E402


class ParseTest(unittest.TestCase):
    def test_seed_specs(self):
        self.assertEqual(parse_seeds("2-4"), [2, 3, 4])
        self.assertEqual(parse_seeds("2,5,7"), [2, 5, 7])
        self.assertEqual(parse_seeds("2-3, 9"), [2, 3, 9])

    def test_lanes(self):
        self.assertEqual(parse_lanes("0:1,1:1,2:1,3:1"), [0, 1, 2, 3])
        self.assertEqual(parse_lanes("0:1,1:2"), [0, 1, 1])

    def test_arm_seeds_unknown_arm_fails_closed(self):
        with self.assertRaises(ValueError):
            parse_arm_seeds("sft_fial=2-7", ["sft_all", "rl_pos", "sft_fail"])

    def test_arm_seeds_parses(self):
        self.assertEqual(parse_arm_seeds("sft_fail=2-7", ["sft_all", "sft_fail"]), {"sft_fail": set(range(2, 8))})
        self.assertEqual(parse_arm_seeds("", ["a"]), {})


class BuildJobsTest(unittest.TestCase):
    ARMS = ["sft_all", "rl_pos", "sft_fail"]

    def test_seed_major_and_interleaved(self):
        jobs = build_jobs(self.ARMS, [2, 3])
        self.assertEqual(jobs, [(2, "sft_all"), (2, "rl_pos"), (2, "sft_fail"),
                                (3, "sft_all"), (3, "rl_pos"), (3, "sft_fail")])

    def test_override_removes_only_the_named_arm_beyond_its_range(self):
        jobs = build_jobs(self.ARMS, list(range(2, 12)), {"sft_fail": set(range(2, 8))})
        self.assertEqual(len(jobs), 10 + 10 + 6)
        self.assertEqual([s for s, a in jobs if a == "sft_fail"], list(range(2, 8)))
        self.assertEqual([s for s, a in jobs if a == "rl_pos"], list(range(2, 12)))
        # order is untouched: still seed-major, arms interleaved within a seed
        self.assertEqual(jobs[:3], [(2, "sft_all"), (2, "rl_pos"), (2, "sft_fail")])
        self.assertEqual(jobs[-2:], [(11, "sft_all"), (11, "rl_pos")])

    def test_no_override_is_identity(self):
        self.assertEqual(build_jobs(["a", "b"], [1, 2]), build_jobs(["a", "b"], [1, 2], {}))


class PerCardUrlTest(unittest.TestCase):
    """--base-url may carry {card}; a plain URL must pass through untouched."""

    def test_placeholder_maps_lane_to_its_own_victim(self):
        url = "http://127.0.0.1:800{card}/v1"
        self.assertEqual([url.format(card=c) for c in parse_lanes("0:1,1:1,2:1,3:1")],
                         ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1",
                          "http://127.0.0.1:8002/v1", "http://127.0.0.1:8003/v1"])

    def test_plain_url_unchanged(self):
        self.assertEqual("http://127.0.0.1:8000/v1".format(card=3), "http://127.0.0.1:8000/v1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
