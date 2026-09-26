"""Fail-closed artifacts for the two-worker V100 Gate 1 execution.

This module is deliberately torch-free.  It owns only the deterministic manifest
partition and the sealed rank-shard/merge envelopes.  Scientific trace
recomputation stays in :mod:`scripts.h1_tooluse_gate1_local`, where the frozen
attacker parser, ReAct parser, domain, and Oracle are available.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .runtime_profile import (
    RUNTIME_PROFILE_SHA256,
    V100_DDP_PROFILE_ID,
    WORKER_VICTIM_PORTS,
    WORLD_SIZE,
)


SHARD_SCHEMA_VERSION = 1
SHARD_KIND = "tooluse_gate1_v100_rank_shard"
MERGE_SCHEMA_VERSION = 1
MERGE_KIND = "tooluse_gate1_v100_rank_merge"
PARTITION_SCHEME = "ordered-manifest-index-mod-world-size-v1"
_GPU_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9][A-Za-z0-9-]{7,127}$")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(document: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(document))
    payload.pop("payload_sha256", None)
    return canonical_sha256(payload)


def seal(document: Mapping[str, Any]) -> dict:
    result = copy.deepcopy(dict(document))
    result.pop("payload_sha256", None)
    result["payload_sha256"] = canonical_sha256(result)
    return result


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"invalid dual-worker Gate 1 artifact: {message}")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_goal_ids(goal_ids: Sequence[str]) -> list[str]:
    _require(
        isinstance(goal_ids, Sequence) and not isinstance(goal_ids, (str, bytes)),
        "goal IDs must be an ordered sequence",
    )
    result = list(goal_ids)
    _require(result and all(isinstance(goal_id, str) and goal_id for goal_id in result),
             "goal IDs must be non-empty strings")
    _require(len(result) == len(set(result)), "goal IDs must be unique")
    return result


def partition_goal_ids(
    goal_ids: Sequence[str], rank: int, *, world_size: int = WORLD_SIZE
) -> list[str]:
    """Return the deterministic, ordered, complete-episode shard for ``rank``."""
    ordered = _validate_goal_ids(goal_ids)
    _require(world_size == WORLD_SIZE, f"world_size must be {WORLD_SIZE}")
    _require(isinstance(rank, int) and not isinstance(rank, bool) and 0 <= rank < world_size,
             f"rank must be in [0,{world_size})")
    return [goal_id for index, goal_id in enumerate(ordered) if index % world_size == rank]


def partition_record(goal_ids: Sequence[str], rank: int) -> dict:
    ordered = _validate_goal_ids(goal_ids)
    assigned = partition_goal_ids(ordered, rank)
    positions = [index for index in range(len(ordered)) if index % WORLD_SIZE == rank]
    return {
        "scheme": PARTITION_SCHEME,
        "world_size": WORLD_SIZE,
        "rank": rank,
        "source_count": len(ordered),
        "source_goal_ids_sha256": canonical_sha256(ordered),
        "assigned_count": len(assigned),
        "assigned_positions": positions,
        "assigned_goal_ids_sha256": canonical_sha256(assigned),
        "assigned_goal_ids": assigned,
    }


def _validate_common_identity(common: Any) -> dict:
    _require(isinstance(common, Mapping), "common_identity must be an object")
    common = copy.deepcopy(dict(common))
    _require(common.get("profile_id") == V100_DDP_PROFILE_ID,
             "common identity is not the V100 DDP profile")
    _require(common.get("profile_sha256") == RUNTIME_PROFILE_SHA256,
             "common identity runtime-profile hash mismatch")
    _require(isinstance(common.get("run_id"), str) and common["run_id"], "run_id missing")
    for key in (
        "deployment", "dataset", "oracle", "models", "victim_request", "restart_proof",
    ):
        _require(isinstance(common.get(key), Mapping), f"common identity {key} missing")
    return common


def worker_endpoint(rank: int) -> str:
    _require(rank in range(WORLD_SIZE), "worker rank out of range")
    return f"http://127.0.0.1:{WORKER_VICTIM_PORTS[rank]}/v1"


def build_worker_identity(
    *,
    rank: int,
    gpu_uuid: str,
    runtime_manifest: Mapping[str, Any],
    victim_service_manifest: Mapping[str, Any],
    attacker_runtime_manifest: Mapping[str, Any],
) -> dict:
    document = {
        "profile_id": V100_DDP_PROFILE_ID,
        "profile_sha256": RUNTIME_PROFILE_SHA256,
        "rank": rank,
        "local_rank": rank,
        "world_size": WORLD_SIZE,
        "gpu_uuid": gpu_uuid,
        "victim_endpoint": worker_endpoint(rank),
        "runtime_manifest": copy.deepcopy(dict(runtime_manifest)),
        "runtime_manifest_sha256": canonical_sha256(runtime_manifest),
        "victim_service_manifest": copy.deepcopy(dict(victim_service_manifest)),
        "victim_service_manifest_sha256": canonical_sha256(victim_service_manifest),
        "attacker_runtime_manifest": copy.deepcopy(dict(attacker_runtime_manifest)),
        "attacker_runtime_manifest_sha256": canonical_sha256(attacker_runtime_manifest),
    }
    return validate_worker_identity(document, require_rank=rank)


def validate_worker_identity(value: Any, *, require_rank: int | None = None) -> dict:
    _require(isinstance(value, Mapping), "worker_identity must be an object")
    worker = copy.deepcopy(dict(value))
    rank = worker.get("rank")
    _require(rank in range(WORLD_SIZE), "worker rank out of range")
    if require_rank is not None:
        _require(rank == require_rank, "worker rank mismatch")
    _require(worker.get("local_rank") == rank and worker.get("world_size") == WORLD_SIZE,
             "worker distributed identity mismatch")
    _require(worker.get("profile_id") == V100_DDP_PROFILE_ID,
             "worker runtime profile mismatch")
    _require(worker.get("profile_sha256") == RUNTIME_PROFILE_SHA256,
             "worker runtime profile hash mismatch")
    _require(isinstance(worker.get("gpu_uuid"), str)
             and _GPU_UUID_RE.fullmatch(worker["gpu_uuid"]) is not None,
             "worker GPU UUID malformed")
    _require(worker.get("victim_endpoint") == worker_endpoint(rank),
             "worker victim endpoint mismatch")
    for key in ("runtime_manifest", "victim_service_manifest", "attacker_runtime_manifest"):
        _require(isinstance(worker.get(key), Mapping), f"worker {key} missing")
        _require(worker.get(f"{key}_sha256") == canonical_sha256(worker[key]),
                 f"worker {key} hash mismatch")
    return worker


def build_rank_shard(
    *,
    common_identity: Mapping[str, Any],
    worker_identity: Mapping[str, Any],
    manifest_goal_ids: Sequence[str],
    tiers: Sequence[str],
    tier_records: Mapping[str, Sequence[Mapping[str, Any]]],
    exit_code: int,
) -> dict:
    common = _validate_common_identity(common_identity)
    worker = validate_worker_identity(worker_identity)
    rank = worker["rank"]
    ordered = _validate_goal_ids(manifest_goal_ids)
    assigned = partition_goal_ids(ordered, rank)
    tier_names = list(tiers)
    _require(tier_names and len(tier_names) == len(set(tier_names)), "tiers malformed")
    _require(set(tier_records) == set(tier_names), "rank shard tier set mismatch")
    normalized_records: dict[str, list[dict]] = {}
    for tier in tier_names:
        records = [copy.deepcopy(dict(record)) for record in tier_records[tier]]
        _require([record.get("goal_id") for record in records] == assigned,
                 f"tier {tier} records differ from assigned manifest order")
        _require(all(record.get("tier") == tier for record in records),
                 f"tier {tier} record label mismatch")
        _require(
            [record.get("episode_index") for record in records]
            == [index for index in range(len(ordered)) if index % WORLD_SIZE == rank],
            f"tier {tier} episode indices are not global manifest positions",
        )
        normalized_records[tier] = records
    _require(isinstance(exit_code, int) and not isinstance(exit_code, bool), "exit_code malformed")
    document = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "kind": SHARD_KIND,
        "status": "complete" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "common_identity": common,
        "common_identity_sha256": canonical_sha256(common),
        "worker_identity": worker,
        "partition": partition_record(ordered, rank),
        "tiers": tier_names,
        "tier_records": normalized_records,
        "tier_records_sha256": {
            tier: canonical_sha256(normalized_records[tier]) for tier in tier_names
        },
    }
    return validate_rank_shard(seal(document), manifest_goal_ids=ordered, tiers=tier_names)


def validate_rank_shard(
    document: Any,
    *,
    manifest_goal_ids: Sequence[str] | None = None,
    tiers: Sequence[str] | None = None,
) -> dict:
    _require(isinstance(document, Mapping), "rank shard must be an object")
    shard = copy.deepcopy(dict(document))
    _require(shard.get("schema_version") == SHARD_SCHEMA_VERSION
             and shard.get("kind") == SHARD_KIND, "rank shard schema/kind mismatch")
    _require(shard.get("payload_sha256") == _payload_sha256(shard),
             "rank shard payload seal mismatch")
    common = _validate_common_identity(shard.get("common_identity"))
    _require(shard.get("common_identity_sha256") == canonical_sha256(common),
             "rank shard common identity hash mismatch")
    worker = validate_worker_identity(shard.get("worker_identity"))
    rank = worker["rank"]
    partition = shard.get("partition")
    _require(isinstance(partition, Mapping), "rank shard partition missing")
    assigned = partition.get("assigned_goal_ids")
    _require(isinstance(assigned, list), "rank shard assigned goals missing")
    if manifest_goal_ids is not None:
        expected_partition = partition_record(manifest_goal_ids, rank)
        _require(dict(partition) == expected_partition, "rank shard partition mismatch")
    else:
        _require(partition.get("scheme") == PARTITION_SCHEME
                 and partition.get("world_size") == WORLD_SIZE
                 and partition.get("rank") == rank,
                 "rank shard partition identity mismatch")
        _require(partition.get("assigned_count") == len(assigned)
                 and partition.get("assigned_goal_ids_sha256") == canonical_sha256(assigned),
                 "rank shard assigned-goal hash/count mismatch")
    tier_names = shard.get("tiers")
    _require(isinstance(tier_names, list) and tier_names, "rank shard tiers missing")
    if tiers is not None:
        _require(tier_names == list(tiers), "rank shard tier order mismatch")
    records_by_tier = shard.get("tier_records")
    record_hashes = shard.get("tier_records_sha256")
    _require(isinstance(records_by_tier, Mapping) and set(records_by_tier) == set(tier_names),
             "rank shard records tier set mismatch")
    _require(isinstance(record_hashes, Mapping) and set(record_hashes) == set(tier_names),
             "rank shard record hashes tier set mismatch")
    expected_positions = partition.get("assigned_positions")
    for tier in tier_names:
        records = records_by_tier[tier]
        _require(isinstance(records, list), f"tier {tier} records malformed")
        _require([record.get("goal_id") for record in records] == assigned,
                 f"tier {tier} rank-shard goal order mismatch")
        _require([record.get("episode_index") for record in records] == expected_positions,
                 f"tier {tier} rank-shard episode positions mismatch")
        _require(all(record.get("tier") == tier for record in records),
                 f"tier {tier} rank-shard labels mismatch")
        _require(record_hashes.get(tier) == canonical_sha256(records),
                 f"tier {tier} rank-shard records hash mismatch")
    _require(shard.get("status") in {"complete", "failed"}, "rank shard status malformed")
    _require(isinstance(shard.get("exit_code"), int)
             and not isinstance(shard.get("exit_code"), bool), "rank shard exit code malformed")
    _require((shard["status"] == "complete") == (shard["exit_code"] == 0),
             "rank shard status/exit code disagree")
    return shard


def load_rank_shard(
    path: str | Path, *, manifest_goal_ids: Sequence[str], tiers: Sequence[str]
) -> dict:
    artifact_path = Path(path)
    _require(artifact_path.is_file(), f"rank shard missing: {artifact_path}")
    try:
        document = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid dual-worker Gate 1 artifact: cannot read shard: {exc}") from exc
    shard = validate_rank_shard(
        document, manifest_goal_ids=manifest_goal_ids, tiers=tiers
    )
    return {
        "path": str(artifact_path.resolve()),
        "file_sha256": file_sha256(artifact_path),
        "payload_sha256": shard["payload_sha256"],
        "rank": shard["worker_identity"]["rank"],
        "document": shard,
    }


def merge_rank_shards(
    shard_refs: Sequence[Mapping[str, Any]],
    *,
    manifest_goal_ids: Sequence[str],
    tiers: Sequence[str],
    recompute_episode: Callable[[str, dict], None],
) -> tuple[dict[str, list[dict]], dict]:
    """Validate exactly two shards, centrally recompute rows, and restore manifest order."""
    ordered = _validate_goal_ids(manifest_goal_ids)
    tier_names = list(tiers)
    _require(len(shard_refs) == WORLD_SIZE, f"exactly {WORLD_SIZE} shard refs required")
    normalized_refs: list[dict] = []
    shards: list[dict] = []
    for raw_ref in shard_refs:
        _require(isinstance(raw_ref, Mapping), "shard ref must be an object")
        ref = copy.deepcopy(dict(raw_ref))
        shard = validate_rank_shard(
            ref.pop("document", None), manifest_goal_ids=ordered, tiers=tier_names
        )
        rank = shard["worker_identity"]["rank"]
        _require(ref.get("rank") == rank, "shard ref rank mismatch")
        _require(_is_sha256(ref.get("file_sha256")), "shard ref file hash malformed")
        _require(ref.get("payload_sha256") == shard["payload_sha256"],
                 "shard ref payload hash mismatch")
        _require(isinstance(ref.get("path"), str) and ref["path"], "shard ref path missing")
        normalized_refs.append(ref)
        shards.append(shard)
    order = sorted(range(WORLD_SIZE), key=lambda index: shards[index]["worker_identity"]["rank"])
    shards = [shards[index] for index in order]
    normalized_refs = [normalized_refs[index] for index in order]
    _require([shard["worker_identity"]["rank"] for shard in shards] == list(range(WORLD_SIZE)),
             "rank shards must be exactly ranks 0 and 1")
    _require(all(shard["status"] == "complete" and shard["exit_code"] == 0 for shard in shards),
             "all rank shards must complete with exit code zero")
    common = shards[0]["common_identity"]
    _require(all(shard["common_identity"] == common for shard in shards[1:]),
             "rank shard common identities differ")
    worker_identities = [shard["worker_identity"] for shard in shards]
    _require(len({worker["gpu_uuid"] for worker in worker_identities}) == WORLD_SIZE,
             "rank shards do not use two distinct GPU UUIDs")
    _require(len({worker["victim_endpoint"] for worker in worker_identities}) == WORLD_SIZE,
             "rank shards do not use two distinct victim endpoints")

    positions = {goal_id: index for index, goal_id in enumerate(ordered)}
    merged: dict[str, list[dict]] = {}
    tier_manifests = []
    for tier in tier_names:
        rows = [copy.deepcopy(record) for shard in shards for record in shard["tier_records"][tier]]
        row_ids = [record.get("goal_id") for record in rows]
        _require(len(rows) == len(ordered), f"tier {tier} merged denominator mismatch")
        _require(len(row_ids) == len(set(row_ids)), f"tier {tier} has duplicate goals")
        _require(set(row_ids) == set(ordered), f"tier {tier} has missing or extra goals")
        rows.sort(key=lambda record: positions[record["goal_id"]])
        _require([record["goal_id"] for record in rows] == ordered,
                 f"tier {tier} cannot restore manifest order")
        for expected_index, record in enumerate(rows):
            _require(record.get("episode_index") == expected_index,
                     f"tier {tier} global episode index mismatch")
            _require(record.get("status") == "valid",
                     f"tier {tier} contains an invalid episode")
            recompute_episode(tier, copy.deepcopy(record))
        merged[tier] = rows
        tier_manifests.append({
            "tier": tier,
            "count": len(rows),
            "goal_ids_sha256": canonical_sha256(ordered),
            "records_sha256": canonical_sha256(rows),
        })

    merge_document = seal({
        "schema_version": MERGE_SCHEMA_VERSION,
        "kind": MERGE_KIND,
        "status": "complete",
        "profile_id": V100_DDP_PROFILE_ID,
        "profile_sha256": RUNTIME_PROFILE_SHA256,
        "world_size": WORLD_SIZE,
        "partition_scheme": PARTITION_SCHEME,
        "common_identity": common,
        "common_identity_sha256": canonical_sha256(common),
        "goal_count": len(ordered),
        "goal_ids_sha256": canonical_sha256(ordered),
        "tiers": tier_manifests,
        "shards": normalized_refs,
        "workers": worker_identities,
    })
    return merged, validate_merge_manifest(
        merge_document, manifest_goal_ids=ordered, tiers=tier_names
    )


def validate_merge_manifest(
    document: Any,
    *,
    manifest_goal_ids: Sequence[str],
    tiers: Sequence[str],
) -> dict:
    """Validate the rank0 merge envelope without trusting its reported counts/hashes."""
    _require(isinstance(document, Mapping), "merge manifest must be an object")
    merged = copy.deepcopy(dict(document))
    _require(merged.get("schema_version") == MERGE_SCHEMA_VERSION
             and merged.get("kind") == MERGE_KIND, "merge manifest schema/kind mismatch")
    _require(merged.get("payload_sha256") == _payload_sha256(merged),
             "merge manifest payload seal mismatch")
    _require(merged.get("status") == "complete", "merge manifest is not complete")
    _require(merged.get("profile_id") == V100_DDP_PROFILE_ID
             and merged.get("profile_sha256") == RUNTIME_PROFILE_SHA256,
             "merge manifest runtime profile mismatch")
    _require(merged.get("world_size") == WORLD_SIZE
             and merged.get("partition_scheme") == PARTITION_SCHEME,
             "merge manifest distributed identity mismatch")
    common = _validate_common_identity(merged.get("common_identity"))
    _require(merged.get("common_identity_sha256") == canonical_sha256(common),
             "merge manifest common identity hash mismatch")
    ordered = _validate_goal_ids(manifest_goal_ids)
    _require(merged.get("goal_count") == len(ordered)
             and merged.get("goal_ids_sha256") == canonical_sha256(ordered),
             "merge manifest goal identity mismatch")
    tier_names = list(tiers)
    tier_manifests = merged.get("tiers")
    _require(isinstance(tier_manifests, list)
             and [item.get("tier") for item in tier_manifests] == tier_names,
             "merge manifest tier order mismatch")
    for item in tier_manifests:
        _require(item.get("count") == len(ordered),
                 f"merge tier {item.get('tier')} denominator mismatch")
        _require(item.get("goal_ids_sha256") == canonical_sha256(ordered),
                 f"merge tier {item.get('tier')} goal hash mismatch")
        _require(_is_sha256(item.get("records_sha256")),
                 f"merge tier {item.get('tier')} records hash malformed")
    refs = merged.get("shards")
    _require(isinstance(refs, list) and len(refs) == WORLD_SIZE,
             "merge manifest must reference exactly two shards")
    _require([ref.get("rank") for ref in refs] == list(range(WORLD_SIZE)),
             "merge manifest shard rank order mismatch")
    for ref in refs:
        _require(isinstance(ref.get("path"), str) and ref["path"],
                 "merge shard path missing")
        _require(_is_sha256(ref.get("file_sha256")) and _is_sha256(ref.get("payload_sha256")),
                 "merge shard hash malformed")
    workers = merged.get("workers")
    _require(isinstance(workers, list) and len(workers) == WORLD_SIZE,
             "merge manifest worker identities missing")
    checked_workers = [validate_worker_identity(worker, require_rank=rank)
                       for rank, worker in enumerate(workers)]
    _require(len({worker["gpu_uuid"] for worker in checked_workers}) == WORLD_SIZE,
             "merge workers do not have distinct GPU UUIDs")
    _require(len({worker["victim_endpoint"] for worker in checked_workers}) == WORLD_SIZE,
             "merge workers do not have distinct victim endpoints")
    return merged
