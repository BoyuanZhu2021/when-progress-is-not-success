"""Pure canonical identity contract for the dual-V100 victim serving pair.

This module is the single implementation used by both the serving manager and
formal evaluation.  It deliberately performs no network, process, or CUDA work.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from .model_pins import (
    REMOTE_HF_HOME,
    VICTIM_HF_MODEL,
    VICTIM_REVISION,
    VICTIM_V100_SERVED_NAME,
)
from .runtime_profile import (
    RUNTIME_PROFILE_SHA256,
    V100_DDP_PROFILE_ID,
    VICTIM_DTYPE,
    VICTIM_QUANTIZATION,
    WORKER_GPU_UUID_ENVS,
    WORKER_VICTIM_PORTS,
    WORLD_SIZE,
    canonical_sha256,
    validate_runtime_manifest,
    validate_worker_inventory,
)


import os
TMP = os.environ.get("OFC_TMP", "/tmp")
REMOTE_CODE = f"{TMP}/h1mt/code"
REMOTE_SRC = f"{REMOTE_CODE}/src"
HOST = "127.0.0.1"
TFSERVE_VERSION = "5.13.1"
TFSERVE_ENV = f"{TMP}/envs/tfserve"
TFSERVE_PYTHON = f"{TFSERVE_ENV}/bin/python"
TFSERVE_BIN = f"{TFSERVE_ENV}/bin/transformers"

PAIR_MANIFEST_PATH = f"{TMP}/h1_victim_v100_pair_manifest.json"
REPLICA_MANIFEST_PATHS = tuple(
    f"{TMP}/h1_victim_v100_replica_{rank}.json" for rank in range(WORLD_SIZE)
)
REPLICA_MANIFEST_KIND = "h1_v100_victim_replica_manifest"
PAIR_MANIFEST_KIND = "h1_v100_victim_pair_manifest"
SCHEMA_VERSION = 1

MODEL_IDENTITY = {
    "hf_model": VICTIM_HF_MODEL,
    "revision": VICTIM_REVISION,
    "served_model": VICTIM_V100_SERVED_NAME,
}
BACKEND_IDENTITY = {
    "name": "transformers-serve",
    "version": TFSERVE_VERSION,
    "dtype": VICTIM_DTYPE,
    "quantization": VICTIM_QUANTIZATION,
    "continuous_batching": False,
}
_PROFILE_ENV = {
    "H1_RUNTIME_PROFILE_ID": V100_DDP_PROFILE_ID,
    "H1_RUNTIME_PROFILE_SHA256": RUNTIME_PROFILE_SHA256,
}
_SERVING_IDENTITY_ENV = {
    "H1_VICTIM_BACKEND": BACKEND_IDENTITY["name"],
    "H1_VICTIM_HF_MODEL": VICTIM_HF_MODEL,
    "H1_VICTIM_REVISION": VICTIM_REVISION,
    "H1_VICTIM_SERVED_MODEL": VICTIM_V100_SERVED_NAME,
    "H1_VICTIM_DTYPE": VICTIM_DTYPE,
    "H1_VICTIM_QUANTIZATION": VICTIM_QUANTIZATION,
    "H1_VICTIM_HOST": HOST,
}
_STATIC_ENV = {
    "PYTHONPATH": REMOTE_SRC,
    "HF_HOME": REMOTE_HF_HOME,
    "HF_HUB_DISABLE_XET": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "OMP_NUM_THREADS": "8",
    **_PROFILE_ENV,
    **_SERVING_IDENTITY_ENV,
}
IDENTITY_ENV_KEYS = tuple(sorted({
    "CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "RANK", "LOCAL_RANK", "WORLD_SIZE",
    "LOCAL_WORLD_SIZE", "H1_VICTIM_RANK", "H1_VICTIM_PORT",
    *WORKER_GPU_UUID_ENVS, *_STATIC_ENV.keys(),
}))

_REPLICA_KEYS = {
    "schema_version", "kind", "path", "profile_id", "profile_sha256",
    "rank", "local_rank", "world_size", "model", "backend",
    "runtime_manifest", "process", "service", "launch_record_sha256",
    "sealed_at", "payload_sha256",
}
_PAIR_KEYS = {
    "schema_version", "kind", "path", "profile_id", "profile_sha256",
    "model", "topology", "replicas", "sealed_at", "payload_sha256",
}


def seal(document: Mapping[str, Any]) -> dict:
    payload = copy.deepcopy(dict(document))
    payload.pop("payload_sha256", None)
    payload["payload_sha256"] = canonical_sha256(payload)
    return payload


def _validate_seal(document: Mapping[str, Any]) -> dict:
    if not isinstance(document, Mapping):
        raise ValueError("sealed document must be a mapping")
    payload = copy.deepcopy(dict(document))
    claimed = payload.pop("payload_sha256", None)
    expected = canonical_sha256(payload)
    if claimed != expected:
        raise ValueError(f"manifest seal mismatch: expected {expected}, got {claimed}")
    payload["payload_sha256"] = claimed
    return payload


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_common(document: Mapping[str, Any], kind: str, keys: set[str]) -> dict:
    result = _validate_seal(document)
    if set(result) != keys:
        raise ValueError(f"{kind} field set mismatch")
    if result.get("schema_version") != SCHEMA_VERSION or result.get("kind") != kind:
        raise ValueError(
            f"unexpected manifest schema/kind: "
            f"{result.get('schema_version')}/{result.get('kind')}"
        )
    if result.get("profile_id") != V100_DDP_PROFILE_ID:
        raise ValueError("manifest is not the active V100 runtime profile")
    if result.get("profile_sha256") != RUNTIME_PROFILE_SHA256:
        raise ValueError("manifest runtime profile hash mismatch")
    if not isinstance(result.get("sealed_at"), str) or not result["sealed_at"]:
        raise ValueError("manifest sealed_at is missing")
    return result


def expected_cmdline(rank: int) -> list[str]:
    if rank not in range(WORLD_SIZE):
        raise ValueError(f"invalid victim rank: {rank!r}")
    return [
        TFSERVE_PYTHON, TFSERVE_BIN, "serve", VICTIM_V100_SERVED_NAME,
        "--host", HOST, "--port", str(WORKER_VICTIM_PORTS[rank]),
        "--device", "cuda:0", "--dtype", VICTIM_DTYPE,
        "--reasoning", "off", "--no-continuous-batching", "--no-compile",
        "--model-timeout", "-1",
    ]


def expected_environment(rank: int, inventory: Mapping[str, Any]) -> dict[str, str]:
    if rank not in range(WORLD_SIZE):
        raise ValueError(f"invalid victim rank: {rank!r}")
    workers = inventory.get("workers") if isinstance(inventory, Mapping) else None
    if not isinstance(workers, Sequence) or len(workers) != WORLD_SIZE:
        raise ValueError("canonical worker inventory is required")
    uuids = [worker["uuid"] for worker in workers]
    result = {
        **_STATIC_ENV,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": uuids[rank],
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(WORLD_SIZE),
        "LOCAL_WORLD_SIZE": str(WORLD_SIZE),
        "H1_VICTIM_RANK": str(rank),
        "H1_VICTIM_PORT": str(WORKER_VICTIM_PORTS[rank]),
    }
    result.update({name: uuids[index] for index, name in enumerate(WORKER_GPU_UUID_ENVS)})
    return dict(sorted(result.items()))


def validate_replica_manifest(
    document: Mapping[str, Any], *, expected_rank: int | None = None,
) -> dict:
    """Validate a sealed replica manifest without touching live process state."""
    result = _validate_common(document, REPLICA_MANIFEST_KIND, _REPLICA_KEYS)
    rank = result.get("rank")
    if rank not in range(WORLD_SIZE) or (expected_rank is not None and rank != expected_rank):
        raise ValueError(f"replica rank mismatch: {rank!r}")
    if result.get("path") != REPLICA_MANIFEST_PATHS[rank]:
        raise ValueError("replica manifest path mismatch")
    if (result.get("local_rank"), result.get("world_size")) != (rank, WORLD_SIZE):
        raise ValueError("replica distributed identity mismatch")
    if result.get("model") != MODEL_IDENTITY or result.get("backend") != BACKEND_IDENTITY:
        raise ValueError("replica model/backend identity mismatch")

    runtime = validate_runtime_manifest(result.get("runtime_manifest"), require_role="worker")
    binding = runtime["binding"]
    if runtime["rank"] != rank or binding["victim_port"] != WORKER_VICTIM_PORTS[rank]:
        raise ValueError("replica runtime rank/port mismatch")
    canonical_inventory = validate_worker_inventory(
        runtime["inventory"], [row["uuid"] for row in runtime["inventory"]]
    )

    process = result.get("process")
    required_process = {
        "pid", "start_time_ticks", "cmdline", "cmdline_sha256", "environ_sha256",
        "environment", "gpu_uuid",
    }
    if not isinstance(process, Mapping) or set(process) != required_process:
        raise ValueError("replica process identity field set mismatch")
    if (not isinstance(process["pid"], int) or process["pid"] < 2
            or not isinstance(process["start_time_ticks"], int)
            or process["start_time_ticks"] < 1):
        raise ValueError("invalid process PID/start ticks")
    cmdline = expected_cmdline(rank)
    if process["cmdline"] != cmdline:
        raise ValueError("process cmdline is not the canonical ordered argv")
    if process["cmdline_sha256"] != canonical_sha256(cmdline):
        raise ValueError("process canonical cmdline hash mismatch")
    if process["environment"] != expected_environment(rank, canonical_inventory):
        raise ValueError("process allowed environment subset is not canonical/exact")
    if not _is_sha256(process["environ_sha256"]):
        raise ValueError("process full environment hash is not a canonical SHA-256")
    if process["gpu_uuid"] != binding["uuid"]:
        raise ValueError("process/runtime GPU UUID mismatch")

    service = result.get("service")
    if (not isinstance(service, Mapping)
            or set(service) != {"endpoint", "port", "chat_response"}
            or service.get("port") != WORKER_VICTIM_PORTS[rank]
            or service.get("endpoint") != f"http://{HOST}:{WORKER_VICTIM_PORTS[rank]}/v1"):
        raise ValueError("replica endpoint mismatch")
    response = service.get("chat_response")
    if (not isinstance(response, Mapping)
            or response.get("model") != VICTIM_V100_SERVED_NAME
            or not isinstance(response.get("choices"), list)
            or not response["choices"]):
        raise ValueError("replica chat proof mismatch")
    if not _is_sha256(result.get("launch_record_sha256")):
        raise ValueError("missing launch-record seal")
    return result


def validate_pair_manifest(
    document: Mapping[str, Any], *, replica_manifests: Sequence[Mapping[str, Any]] | None = None,
) -> dict:
    """Validate the aggregate seal and its exact two canonical replica references."""
    result = _validate_common(document, PAIR_MANIFEST_KIND, _PAIR_KEYS)
    if result.get("path") != PAIR_MANIFEST_PATH or result.get("model") != MODEL_IDENTITY:
        raise ValueError("pair path/model identity mismatch")
    if result.get("topology") != {
        "world_size": WORLD_SIZE, "ports": list(WORKER_VICTIM_PORTS),
    }:
        raise ValueError("pair topology mismatch")
    if replica_manifests is None or len(replica_manifests) != WORLD_SIZE:
        raise ValueError("both canonical replica manifests are required for pair validation")
    replicas = [
        validate_replica_manifest(replica_manifests[rank], expected_rank=rank)
        for rank in range(WORLD_SIZE)
    ]
    expected_refs = [
        {
            "rank": rank,
            "port": WORKER_VICTIM_PORTS[rank],
            "path": REPLICA_MANIFEST_PATHS[rank],
            "payload_sha256": replicas[rank]["payload_sha256"],
        }
        for rank in range(WORLD_SIZE)
    ]
    if result.get("replicas") != expected_refs:
        raise ValueError("pair replica reference mismatch")
    return result

