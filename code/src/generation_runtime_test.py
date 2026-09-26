from __future__ import annotations

import unittest

from .generation_runtime import (
    CACHED_EVAL_GENERATION,
    InProcessPEFTCanaryError,
    ReservedTagDecoderGuard,
    ReservedTagDecoderGuardError,
    audit_inprocess_peft_canary,
    cached_eval_generation,
    cached_eval_generation_identity,
    inprocess_output_fingerprint,
    reserved_tag_case_variants,
)


class _FakeModel:
    def __init__(self, training: bool):
        self.training = training
        self.calls: list[tuple[str, bool | None]] = []

    def eval(self):
        self.calls.append(("eval", None))
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.calls.append(("train", mode))
        self.training = mode
        return self


class _FakeTokenizer:
    name_or_path = "fake/qwen"
    special_tokens_map = {"eos_token": "<eos>"}

    def __init__(self):
        self.tokens = [
            "safe", "<inject>", "<inj", "ect>\"", "</INJECT>)", ">", "<", "INJECT",
        ]

    def __len__(self):
        return len(self.tokens)

    def decode(self, token_ids, **_kwargs):
        return "".join(self.tokens[token_id] for token_id in token_ids)

    def batch_decode(self, sequences, **kwargs):
        return [self.decode(sequence, **kwargs) for sequence in sequences]


class _FakeInputIds:
    def __init__(self, rows):
        self._rows = rows

    def tolist(self):
        return [list(row) for row in self._rows]


class _FakeScores:
    def __init__(self, rows: int, columns: int):
        self.values = [[0.0] * columns for _ in range(rows)]

    def __setitem__(self, key, value):
        row, columns = key
        for column in columns:
            self.values[row][column] = value

class CachedEvalGenerationTest(unittest.TestCase):
    def test_restores_training_mode_even_on_error(self):
        model = _FakeModel(training=True)
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with cached_eval_generation(model):
                self.assertFalse(model.training)
                raise RuntimeError("boom")
        self.assertTrue(model.training)
        self.assertEqual(model.calls, [("eval", None), ("train", True)])

    def test_restores_prior_eval_mode_and_identity_is_copy(self):
        model = _FakeModel(training=False)
        with cached_eval_generation(model):
            self.assertFalse(model.training)
        self.assertFalse(model.training)
        identity = cached_eval_generation_identity()
        identity["use_cache"] = False
        self.assertTrue(CACHED_EVAL_GENERATION["use_cache"])

    def test_inprocess_fingerprint_and_a_b_a_audit(self):
        original = inprocess_output_fingerprint({
            "text": "A", "prompt_ids": [1, 2], "resp_ids": [3],
        })
        mutated = inprocess_output_fingerprint({
            "text": "B", "prompt_ids": [1, 2], "resp_ids": [4],
        })
        request = "a" * 64
        audit = audit_inprocess_peft_canary(
            serial_fingerprints=[original] * 4,
            mutated_fingerprint=mutated,
            restored_fingerprint=original,
            request_sha256s=[request] * 6,
            lora_parameter_sha256s=["1" * 64, "2" * 64, "1" * 64],
            training_mode_restored=[True] * 6,
        )
        self.assertEqual(audit["status"], "PASS")
        self.assertTrue(audit["mutation_changed_behavior"])

    def test_inprocess_audit_rejects_behaviorally_inert_mutation(self):
        fingerprint = "f" * 64
        with self.assertRaisesRegex(
            InProcessPEFTCanaryError, "mutated B did not change"
        ):
            audit_inprocess_peft_canary(
                serial_fingerprints=[fingerprint] * 4,
                mutated_fingerprint=fingerprint,
                restored_fingerprint=fingerprint,
                request_sha256s=["a" * 64] * 6,
                lora_parameter_sha256s=["1" * 64, "2" * 64, "1" * 64],
                training_mode_restored=[True] * 6,
            )

    def test_reserved_tag_variants_are_exactly_128(self):
        variants = reserved_tag_case_variants()
        self.assertEqual(len(variants), 128)
        self.assertEqual(len(set(variants)), 128)
        self.assertIn("<inject>", variants)
        self.assertIn("</INJECT>", variants)

    def test_text_aware_guard_blocks_fused_and_cross_token_completions(self):
        guard = ReservedTagDecoderGuard(_FakeTokenizer(), tokenizer_revision="revision-1")
        self.assertEqual(guard.identity()["case_variant_count"], 128)
        for variant in reserved_tag_case_variants():
            with self.assertRaises(ReservedTagDecoderGuardError):
                guard.state_for_text("prefix " + variant + " suffix")
        self.assertIn(1, guard.blocked_token_ids(""))
        self.assertIn(4, guard.blocked_token_ids(""))
        self.assertEqual(guard.state_for_token_ids([2]), "<inj")
        self.assertIn(3, guard.blocked_token_ids("<inj"))
        with self.assertRaisesRegex(ReservedTagDecoderGuardError, "already contains"):
            guard.state_for_text("prefix <InJeCt> suffix")

        scores = _FakeScores(rows=2, columns=len(_FakeTokenizer()))
        guard.bind(prompt_length=2)(
            _FakeInputIds([[99, 99, 2], [99, 99, 0]]), scores,
        )
        self.assertEqual(scores.values[0][3], -float("inf"))
        self.assertEqual(scores.values[1][1], -float("inf"))
        self.assertEqual(guard.metrics()["rows_checked"], 2)


if __name__ == "__main__":
    unittest.main()
