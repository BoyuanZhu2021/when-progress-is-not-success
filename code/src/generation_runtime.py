"""Explicit attacker generation phase controls for QLoRA rollouts.

The training model keeps gradient checkpointing enabled for policy-gradient backward passes.
Autoregressive rollout generation must nevertheless run in evaluation mode with KV caching;
otherwise Transformers disables the cache for a training-mode checkpointed model and repeatedly
recomputes the growing prefix.  The context manager restores the exact prior training state.
"""
from __future__ import annotations

import contextlib
import copy
from typing import Any, Iterator

from .victim_decision_protocol import canonical_sha256


CACHED_EVAL_GENERATION_ID = "h1-attacker-cached-eval-generation-v1"
CACHED_EVAL_GENERATION = {
    "schema_version": 1,
    "runtime_id": CACHED_EVAL_GENERATION_ID,
    "model_mode_during_generate": "eval",
    "use_cache": True,
    "gradient_checkpointing_during_backward": True,
    "restore_prior_training_mode": True,
    "release_reserved_cuda_memory_before_victim": True,
    "sampling_semantics": "unchanged-seed-temperature-top-p-max-new-tokens",
}
CACHED_EVAL_GENERATION["payload_sha256"] = canonical_sha256(CACHED_EVAL_GENERATION)

RESERVED_TAG_DECODER_GUARD_ID = "h1-content-only-reserved-tag-dfa-v1"
_CANONICAL_RESERVED_TAGS = ("<inject>", "</inject>")


class ReservedTagDecoderGuardError(RuntimeError):
    """The constrained decoder contract could not be constructed or was violated."""


def reserved_tag_case_variants() -> tuple[str, ...]:
    """Return all 128 ASCII-case variants of the two reserved transport tags."""
    words = []
    for mask in range(1 << len("inject")):
        words.append("".join(
            character.upper() if mask & (1 << index) else character.lower()
            for index, character in enumerate("inject")
        ))
    return tuple(sorted({
        tag
        for word in words
        for tag in (f"<{word}>", f"</{word}>")
    }))


def _reserved_tag_states() -> tuple[str, ...]:
    states = {""}
    for pattern in _CANONICAL_RESERVED_TAGS:
        states.update(pattern[:length] for length in range(1, len(pattern)))
    return tuple(sorted(states, key=lambda value: (len(value), value)))


_RESERVED_TAG_STATES = _reserved_tag_states()


def _advance_reserved_tag_state(state: str, fragment: str) -> str | None:
    """Advance the case-insensitive DFA; ``None`` means a tag was completed."""
    combined = (state + fragment).lower()
    if any(pattern in combined for pattern in _CANONICAL_RESERVED_TAGS):
        return None
    return max(
        (prefix for prefix in _RESERVED_TAG_STATES if combined.endswith(prefix)),
        key=len,
    )


def _decode_ids(tokenizer: Any, token_ids: list[int]) -> str:
    kwargs = {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
    try:
        value = tokenizer.decode(token_ids, **kwargs)
    except TypeError:
        kwargs.pop("clean_up_tokenization_spaces")
        value = tokenizer.decode(token_ids, **kwargs)
    if not isinstance(value, str):
        raise ReservedTagDecoderGuardError("tokenizer.decode did not return text")
    return value


def _token_texts(tokenizer: Any, vocab_size: int) -> list[str]:
    """Decode every candidate token once, using bounded batches when supported."""
    result: list[str] = []
    batch_decode = getattr(tokenizer, "batch_decode", None)
    if callable(batch_decode):
        for start in range(0, vocab_size, 4096):
            sequences = [[token_id] for token_id in range(start, min(vocab_size, start + 4096))]
            kwargs = {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
            try:
                batch = batch_decode(sequences, **kwargs)
            except TypeError:
                kwargs.pop("clean_up_tokenization_spaces")
                batch = batch_decode(sequences, **kwargs)
            if len(batch) != len(sequences) or any(not isinstance(value, str) for value in batch):
                raise ReservedTagDecoderGuardError("tokenizer.batch_decode returned an invalid batch")
            result.extend(batch)
    else:
        result.extend(_decode_ids(tokenizer, [token_id]) for token_id in range(vocab_size))
    return result


class ReservedTagDecoderGuard:
    """Text-aware DFA that masks tokens before they complete a reserved transport tag.

    A token can contain several decoded characters (for example ``ect>\"``), so blocking fixed
    token-ID sequences is insufficient.  This guard decodes each vocabulary token once and builds
    a state-specific mask over the case-insensitive text DFA.  The parser remains an independent
    fail-closed check after raw output is durably recorded.
    """

    def __init__(self, tokenizer: Any, *, tokenizer_revision: str):
        if not isinstance(tokenizer_revision, str) or not tokenizer_revision:
            raise ReservedTagDecoderGuardError("tokenizer revision must be sealed")
        try:
            vocab_size = len(tokenizer)
        except (TypeError, AttributeError) as exc:
            raise ReservedTagDecoderGuardError("tokenizer has no finite vocabulary size") from exc
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 1:
            raise ReservedTagDecoderGuardError("tokenizer vocabulary size is invalid")
        token_texts = _token_texts(tokenizer, vocab_size)
        blocked = {state: [] for state in _RESERVED_TAG_STATES}
        for token_id, fragment in enumerate(token_texts):
            for state in _RESERVED_TAG_STATES:
                if _advance_reserved_tag_state(state, fragment) is None:
                    blocked[state].append(token_id)
        if not blocked["<inject"] or not blocked["</inject"]:
            raise ReservedTagDecoderGuardError(
                "tokenizer exposes no token that completes a reserved-tag prefix"
            )
        variants = reserved_tag_case_variants()
        table_payload = {state: ids for state, ids in blocked.items()}
        special_tokens = {
            str(key): str(value)
            for key, value in sorted((getattr(tokenizer, "special_tokens_map", {}) or {}).items())
        }
        identity = {
            "schema_version": 1,
            "guard_id": RESERVED_TAG_DECODER_GUARD_ID,
            "scope": "attacker-evaluation-generation-only",
            "matching": "ascii-case-insensitive-equivalent-to-128-explicit-variants",
            "canonical_patterns": list(_CANONICAL_RESERVED_TAGS),
            "case_variant_count": len(variants),
            "case_variants_sha256": canonical_sha256(list(variants)),
            "algorithm": "decoded-text-aho-prefix-dfa-mask-before-sampling-v1",
            "post_generation_repair": False,
            "parser_remains_fail_closed": True,
            "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "")),
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_revision": tokenizer_revision,
            "tokenizer_vocab_size": vocab_size,
            "tokenizer_special_tokens_sha256": canonical_sha256(special_tokens),
            "dfa_states": list(_RESERVED_TAG_STATES),
            "blocked_token_counts_by_state": {
                state: len(ids) for state, ids in blocked.items()
            },
            "transition_table_sha256": canonical_sha256(table_payload),
        }
        identity["payload_sha256"] = canonical_sha256(identity)
        self._tokenizer = tokenizer
        self._blocked = {state: tuple(ids) for state, ids in blocked.items()}
        self._identity = identity
        self._metrics = {
            "logits_calls": 0,
            "rows_checked": 0,
            "rows_with_nonempty_prefix_state": 0,
            "masked_candidate_slots": 0,
        }

    def identity(self) -> dict:
        return copy.deepcopy(self._identity)

    def metrics(self) -> dict:
        return copy.deepcopy(self._metrics)

    def state_for_text(self, text: str) -> str:
        if not isinstance(text, str):
            raise ReservedTagDecoderGuardError("generated content must decode to text")
        state = _advance_reserved_tag_state("", text)
        if state is None:
            raise ReservedTagDecoderGuardError("generated prefix already contains a reserved tag")
        return state

    def state_for_token_ids(self, token_ids: list[int]) -> str:
        return self.state_for_text(_decode_ids(self._tokenizer, token_ids))

    def blocked_token_ids(self, state: str) -> tuple[int, ...]:
        if state not in self._blocked:
            raise ReservedTagDecoderGuardError("unknown reserved-tag DFA state")
        return self._blocked[state]

    def bind(self, prompt_length: int) -> "_BoundReservedTagLogitsProcessor":
        if isinstance(prompt_length, bool) or not isinstance(prompt_length, int) or prompt_length < 1:
            raise ReservedTagDecoderGuardError("generation prompt length is invalid")
        return _BoundReservedTagLogitsProcessor(self, prompt_length)

    def _apply(self, input_ids: Any, scores: Any, *, prompt_length: int) -> Any:
        if hasattr(input_ids, "detach"):
            rows = input_ids.detach().cpu().tolist()
        elif hasattr(input_ids, "tolist"):
            rows = input_ids.tolist()
        else:
            rows = list(input_ids)
        if not isinstance(rows, list):
            raise ReservedTagDecoderGuardError("generation input IDs are not batched")
        self._metrics["logits_calls"] += 1
        for row_index, row in enumerate(rows):
            if not isinstance(row, list) or len(row) < prompt_length:
                raise ReservedTagDecoderGuardError("generation row is shorter than its prompt")
            state = self.state_for_token_ids([int(token_id) for token_id in row[prompt_length:]])
            blocked = self._blocked[state]
            self._metrics["rows_checked"] += 1
            if state:
                self._metrics["rows_with_nonempty_prefix_state"] += 1
            if blocked:
                scores[row_index, list(blocked)] = -float("inf")
                self._metrics["masked_candidate_slots"] += len(blocked)
        return scores


class _BoundReservedTagLogitsProcessor:
    def __init__(self, guard: ReservedTagDecoderGuard, prompt_length: int):
        self._guard = guard
        self._prompt_length = prompt_length

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        return self._guard._apply(input_ids, scores, prompt_length=self._prompt_length)


class InProcessPEFTCanaryError(RuntimeError):
    """The direct in-process PEFT behavioral gate failed closed."""


@contextlib.contextmanager
def cached_eval_generation(model: Any) -> Iterator[None]:
    """Temporarily enter eval mode and restore the caller's exact train/eval state."""
    was_training = bool(model.training)
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def cached_eval_generation_identity() -> dict:
    return copy.deepcopy(CACHED_EVAL_GENERATION)


def inprocess_output_fingerprint(generated: dict[str, Any]) -> str:
    """Hash exact prompt tokens, response tokens and decoded text from model.generate."""
    if not isinstance(generated, dict):
        raise ValueError("in-process generation item must be a mapping")

    def token_ids(field: str) -> list[int]:
        value = generated.get(field)
        if hasattr(value, "detach"):
            value = value.detach().cpu().tolist()
        elif hasattr(value, "tolist"):
            value = value.tolist()
        if (not isinstance(value, list) or not value
                or any(not isinstance(token, int) for token in value)):
            raise ValueError(f"in-process {field} must be non-empty integer token IDs")
        return value

    text = generated.get("text")
    if not isinstance(text, str):
        raise ValueError("in-process generated text must be a string")
    return canonical_sha256({
        "prompt_ids": token_ids("prompt_ids"),
        "resp_ids": token_ids("resp_ids"),
        "text": text,
    })


def audit_inprocess_peft_canary(
    *, serial_fingerprints: list[str], mutated_fingerprint: str,
    restored_fingerprint: str, request_sha256s: list[str],
    lora_parameter_sha256s: list[str], training_mode_restored: list[bool],
) -> dict[str, Any]:
    """Require deterministic A, changed B, exact restored A and exact PEFT identities."""
    if len(serial_fingerprints) < 3:
        raise InProcessPEFTCanaryError("in-process canary needs at least three A repeats")
    fingerprints = [*serial_fingerprints, mutated_fingerprint, restored_fingerprint]
    if any(not isinstance(value, str) or len(value) != 64 for value in fingerprints):
        raise InProcessPEFTCanaryError("in-process output fingerprint is malformed")
    if len(set(serial_fingerprints)) != 1:
        raise InProcessPEFTCanaryError("in-process original A is not exactly reproducible")
    if mutated_fingerprint == serial_fingerprints[0]:
        raise InProcessPEFTCanaryError("in-process mutated B did not change generation")
    if restored_fingerprint != serial_fingerprints[0]:
        raise InProcessPEFTCanaryError("in-process restored A did not restore generation")
    if (len(request_sha256s) != len(fingerprints)
            or any(not isinstance(value, str) or len(value) != 64
                   for value in request_sha256s)
            or len(set(request_sha256s)) != 1):
        raise InProcessPEFTCanaryError("in-process canary request identities differ")
    if (len(lora_parameter_sha256s) != 3
            or lora_parameter_sha256s[0] != lora_parameter_sha256s[2]
            or lora_parameter_sha256s[0] == lora_parameter_sha256s[1]):
        raise InProcessPEFTCanaryError("in-process A-B-A LoRA parameter identity is invalid")
    if (len(training_mode_restored) != len(fingerprints)
            or not all(value is True for value in training_mode_restored)):
        raise InProcessPEFTCanaryError("cached generation did not restore training mode")
    return {
        "status": "PASS",
        "serial_repeats": len(serial_fingerprints),
        "request_sha256": request_sha256s[0],
        "original_output_fingerprint": serial_fingerprints[0],
        "mutated_output_fingerprint": mutated_fingerprint,
        "restored_output_fingerprint": restored_fingerprint,
        "serial_exact": True,
        "mutation_changed_behavior": True,
        "restore_exact": True,
        "training_mode_restored": True,
    }


__all__ = [
    "CACHED_EVAL_GENERATION",
    "CACHED_EVAL_GENERATION_ID",
    "InProcessPEFTCanaryError",
    "RESERVED_TAG_DECODER_GUARD_ID",
    "ReservedTagDecoderGuard",
    "ReservedTagDecoderGuardError",
    "audit_inprocess_peft_canary",
    "cached_eval_generation",
    "cached_eval_generation_identity",
    "inprocess_output_fingerprint",
    "reserved_tag_case_variants",
]
