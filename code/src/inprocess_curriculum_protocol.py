"""CPU-only protocol for the efficiency-first partial-reachable H1 pilot."""
from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .runtime_profile import LEGACY_H20_PROFILE_ID
from .tooluse_gate1_spec import load_frozen_gate1, sha256_file


PROFILE_ID = "h1-partial-reachable-inprocess-pilot-v1"
CONFIG_KIND = "h1_partial_reachable_curriculum_config"
RUN_KIND = "h1_partial_reachable_inprocess_run"
RESULT_KIND = "h1_partial_reachable_inprocess_result"
VERDICT_KIND = "h1_partial_reachable_preliminary_verdict"
AUTHORIZED_INSTANCE = "fa85409945-b6dee8ab"
# Instances that ran earlier campaigns.  Their configs stay loadable so sealed evidence can
# be re-checked, but only `AUTHORIZED_INSTANCE` passes the runtime hardware gate in
# `h1_inprocess_curriculum_pilot._hardware_identity`, so a retired config cannot launch.
RETIRED_INSTANCES = ("20d84f9474-d7816b14",)
KNOWN_INSTANCES = (AUTHORIZED_INSTANCE, *RETIRED_INSTANCES)
ALLOWED_ARMS = ("base", "dense", "sparse")
ALLOWED_SEEDS = (0, 1, 2)
HISTORICAL_VARIANT = "historical-v1"
GATE_PARTIAL_VARIANT = "gate-partial-targeted-v1"
GATE_PARTIAL_NONE_VARIANT = "gate-partial-none-targeted-v1"
GATE_PARTIAL_LEGACY_VARIANT = "gate-partial-legacy-targeted-v1"
GATE_PARTIAL_LEGACY_CONFIRMATORY_VARIANT = "gate-partial-legacy-confirmatory-v1"
LOCAL_COMPACT_PROTOCOL_ID = "h1-local-victim-one-decision-compact-terminal-v1"
LEGACY_PROTOCOL_ID = "h1-victim-one-decision-step-bound-observation-ref-v3"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def goal_ids_sha256(goal_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(goal_ids).encode("utf-8")).hexdigest()


def seal_payload(value: Mapping[str, Any]) -> dict:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise ValueError("payload is already sealed")
    result["payload_sha256"] = canonical_sha256(result)
    return result


# The Gate-1 PASS artifact that selected the light defense tier lived only on the retired
# instance `20d84f9474-d7816b14` and went away with it. Its entire scientific content is one
# frozen decision — defense tier = light — which the PI made on 2026-07-10 and which is
# already declared by every campaign recipe. Re-running the sweep would not restore the
# original provenance, only re-derive the same constant, so a campaign may instead cite the
# decision directly. The tier itself is unchanged and still enforced.
REGISTERED_DEFENSE_SELECTION = "registered:light-gate1-2026-07-10"
FROZEN_DEFENSE_TIER = "light"
DEFENSE_SELECTION_EXP = "EXP-2026W28-007"


def build_registered_defense_selection() -> dict:
    """Frozen light-tier selection, sourced from the registered decision rather than a file."""
    return seal_payload({
        "schema_version": 1,
        "kind": "h1_api_victim_defense_selection",
        "selected_tier": FROZEN_DEFENSE_TIER,
        "source": "registered_constant",
        "source_decision_exp": DEFENSE_SELECTION_EXP,
        "source_decision_date": "2026-07-10",
        "source_gate_file": None,
        "source_gate_file_sha256": None,
        "retired_gate_instance": RETIRED_INSTANCES[0],
        "transfer_scope": "defense_selection_only",
        "api_victim_gate_claim": False,
    })


def build_gate_selection(path: str | Path) -> dict:
    """Bind the current curriculum to the frozen light defense tier."""
    if str(path) == REGISTERED_DEFENSE_SELECTION:
        return build_registered_defense_selection()
    gate_path = Path(path).resolve()
    gate = load_frozen_gate1(
        gate_path,
        expected_profile_id=LEGACY_H20_PROFILE_ID,
        verify_external_artifacts=False,
    )
    if gate.get("verdict") != "PASS" or gate.get("passed") is not True:
        raise ValueError("defense selection source is not a PASS Gate")
    if gate.get("chosen_tier") != "light":
        raise ValueError("in-process profile is frozen to the selected light defense")
    calibration = gate.get("calibration") or {}
    if calibration.get("count") != 69:
        raise ValueError("defense selection does not cover 69 calibration goals")
    return seal_payload({
        "schema_version": 1,
        "kind": "h1_api_victim_defense_selection",
        "selected_tier": "light",
        "source_gate_run_id": gate.get("run_id"),
        "source_gate_profile_id": LEGACY_H20_PROFILE_ID,
        "source_gate_file": str(gate_path),
        "source_gate_file_sha256": sha256_file(gate_path),
        "source_gate_canonical_sha256": canonical_sha256(gate),
        "calibration_goal_ids_sha256": calibration.get("goal_ids_sha256"),
        "transfer_scope": "defense_selection_only",
        "api_victim_gate_claim": False,
    })


def validate_seal(value: Mapping[str, Any]) -> dict:
    result = copy.deepcopy(dict(value))
    claimed = result.pop("payload_sha256", None)
    if claimed != canonical_sha256(result):
        raise ValueError("payload seal mismatch")
    result["payload_sha256"] = claimed
    return result


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("profile_id") != PROFILE_ID:
        raise ValueError("curriculum profile mismatch")
    if value.get("decision_bearing") is not False:
        raise ValueError("curriculum must be explicitly non-decision-bearing")
    if value.get("instance_id") not in KNOWN_INSTANCES:
        raise ValueError("curriculum instance mismatch")
    data = value.get("data") or {}
    train = data.get("training_goal_ids")
    heldout = data.get("heldout_goal_ids")
    if not isinstance(train, list) or not isinstance(heldout, list):
        raise ValueError("curriculum goal lists are missing")
    variant = value.get("curriculum_variant", HISTORICAL_VARIANT)
    expected_shape = {
        HISTORICAL_VARIANT: {"train": 12, "heldout": 12, "seeds": [0, 1], "steps": 12},
        GATE_PARTIAL_VARIANT: {"train": 8, "heldout": 12, "seeds": [0], "steps": 8},
        GATE_PARTIAL_NONE_VARIANT: {
            "train": 8, "heldout": 12, "seeds": [0], "steps": 8,
        },
        GATE_PARTIAL_LEGACY_VARIANT: {
            "train": 8, "heldout": 12, "seeds": [0], "steps": 8,
        },
        GATE_PARTIAL_LEGACY_CONFIRMATORY_VARIANT: {
            "train": 8, "heldout": 12, "seeds": [0, 1, 2], "steps": 8,
        },
    }.get(variant)
    if expected_shape is None:
        raise ValueError("unsupported curriculum variant")
    expected_split = (
        "calibration"
        if variant in (
            GATE_PARTIAL_VARIANT,
            GATE_PARTIAL_NONE_VARIANT,
            GATE_PARTIAL_LEGACY_VARIANT,
            GATE_PARTIAL_LEGACY_CONFIRMATORY_VARIANT,
        )
        else "train"
    )
    if data.get("source_split", "train") != expected_split:
        raise ValueError("curriculum source split mismatch")
    if len(train) != expected_shape["train"] or len(heldout) != expected_shape["heldout"]:
        raise ValueError("curriculum train/held-out shape mismatch")
    if len(set(train)) != len(train) or len(set(heldout)) != len(heldout):
        raise ValueError("curriculum goal IDs are not unique")
    if set(train) & set(heldout):
        raise ValueError("curriculum train and held-out goals overlap")
    if data.get("training_goal_ids_sha256") != goal_ids_sha256(train):
        raise ValueError("curriculum train goal hash mismatch")
    if data.get("heldout_goal_ids_sha256") != goal_ids_sha256(heldout):
        raise ValueError("curriculum held-out goal hash mismatch")
    training = value.get("training") or {}
    interaction = value.get("interaction") or {}
    evaluation = value.get("evaluation") or {}
    exact = {
        "seeds": expected_shape["seeds"], "arms": ["dense", "sparse"],
        "steps_per_arm": expected_shape["steps"], "goals_per_step": 4,
        "trajectories_per_goal": 8, "goal_appearances_per_run": 4,
    }
    for key, expected in exact.items():
        if training.get(key) != expected:
            raise ValueError(f"curriculum training drift: {key}")
    for key, expected in {"T": 5, "max_calls": 3, "max_new_tokens": 256}.items():
        if interaction.get(key) != expected:
            raise ValueError(f"curriculum interaction drift: {key}")
    if interaction.get("defense_tier") not in {"none", "light"}:
        raise ValueError("curriculum defense tier is unsupported")
    expected_defense = "none" if variant == GATE_PARTIAL_NONE_VARIANT else "light"
    if interaction.get("defense_tier") != expected_defense:
        raise ValueError("curriculum defense tier/variant mismatch")
    expected_protocol = (
        LEGACY_PROTOCOL_ID
        if variant in (
            GATE_PARTIAL_LEGACY_VARIANT,
            GATE_PARTIAL_LEGACY_CONFIRMATORY_VARIANT,
        )
        else LOCAL_COMPACT_PROTOCOL_ID
    )
    if interaction.get("victim_decision_protocol_id") != expected_protocol:
        raise ValueError("curriculum victim decision protocol/variant mismatch")
    expected_strict_arguments = variant in (
        GATE_PARTIAL_LEGACY_VARIANT,
        GATE_PARTIAL_LEGACY_CONFIRMATORY_VARIANT,
    )
    if (
        interaction.get("strict_declared_action_arguments")
        is not expected_strict_arguments
    ):
        raise ValueError("curriculum strict action-argument policy/variant mismatch")
    expected_string_bound = 512 if variant in (
        GATE_PARTIAL_LEGACY_VARIANT,
        GATE_PARTIAL_LEGACY_CONFIRMATORY_VARIANT,
    ) else None
    for key in ("final_answer_max_length", "action_string_max_length"):
        if interaction.get(key) != expected_string_bound:
            raise ValueError(f"curriculum victim string bound/variant mismatch: {key}")
    if evaluation.get("samples_per_goal") != 4 or evaluation.get("final_ood_read") is not False:
        raise ValueError("curriculum evaluation drift")
    value["config_file_sha256"] = file_sha256(config_path)
    return value


def build_balanced_schedule(*, seed: int, n_goals: int = 12) -> list[list[int]]:
    if seed not in ALLOWED_SEEDS or n_goals not in (8, 12):
        raise ValueError("unsupported curriculum schedule request")
    flat: list[int] = []
    for epoch in range(4):
        indices = list(range(n_goals))
        derived = int.from_bytes(
            hashlib.sha256(f"h1-curriculum-v1:{seed}:{epoch}".encode()).digest()[:8],
            "big",
        )
        random.Random(derived).shuffle(indices)
        flat.extend(indices)
    schedule = [flat[offset:offset + 4] for offset in range(0, len(flat), 4)]
    if len(schedule) != n_goals or any(len(set(row)) != 4 for row in schedule):
        raise RuntimeError("curriculum schedule construction invariant failed")
    if Counter(flat) != Counter({index: 4 for index in range(n_goals)}):
        raise RuntimeError("curriculum schedule is not balanced")
    return schedule


def build_run_config(
    *, config: Mapping[str, Any], arm: str, seed: int, tag: str,
    gpu_uuid: str, deployment_tree_sha256: str,
    service_manifest_payload_sha256: str, gate_selection_payload_sha256: str,
    initial_lora_sha256: str,
) -> dict:
    if arm not in ALLOWED_ARMS or seed not in ALLOWED_SEEDS:
        raise ValueError("pilot arm/seed is not registered")
    if not tag or not gpu_uuid.startswith("GPU-"):
        raise ValueError("pilot runtime identity is malformed")
    for digest in (
        deployment_tree_sha256, service_manifest_payload_sha256,
        gate_selection_payload_sha256, initial_lora_sha256,
    ):
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("pilot identity digest is malformed")
    return seal_payload({
        "schema_version": 1,
        "kind": RUN_KIND,
        "profile_id": PROFILE_ID,
        "curriculum_variant": config.get("curriculum_variant", HISTORICAL_VARIANT),
        "decision_bearing": False,
        "arm": arm,
        "seed": seed,
        "tag": tag,
        "instance_id": AUTHORIZED_INSTANCE,
        "gpu_uuid": gpu_uuid,
        "config_file_sha256": config["config_file_sha256"],
        "training_goal_ids_sha256": config["data"]["training_goal_ids_sha256"],
        "heldout_goal_ids_sha256": config["data"]["heldout_goal_ids_sha256"],
        "defense_tier": config["interaction"]["defense_tier"],
        "victim_decision_protocol_id": config["interaction"][
            "victim_decision_protocol_id"
        ],
        "strict_declared_action_arguments": config["interaction"][
            "strict_declared_action_arguments"
        ],
        "final_answer_max_length": config["interaction"].get(
            "final_answer_max_length"
        ),
        "action_string_max_length": config["interaction"].get(
            "action_string_max_length"
        ),
        "goal_schedule": None if arm == "base" else build_balanced_schedule(
            seed=seed, n_goals=len(config["data"]["training_goal_ids"])
        ),
        "deployment_tree_sha256": deployment_tree_sha256,
        "service_manifest_payload_sha256": service_manifest_payload_sha256,
        "gate_selection_payload_sha256": gate_selection_payload_sha256,
        "initial_lora_sha256": initial_lora_sha256,
    })
