"""Torch-free runtime profiles and GPU binding for H1.

The active formal profile is one H20 with an FP8-only victim and an NF4/BF16
attacker.  V100 definitions remain solely so historical artifacts can be read;
the active H20 workflow never dispatches to them.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any


RUNTIME_PROFILE_SCHEMA_VERSION = 2
H20_PROFILE_ID = "h20-single-gpu-fp8-only-v2"
# Compatibility import name used throughout the existing H20 implementation.
LEGACY_H20_PROFILE_ID = H20_PROFILE_ID
V100_DDP_PROFILE_ID = "dual-v100-sxm2-32gb-colocated-ddp-v1"
RUNTIME_PROFILE_ID = H20_PROFILE_ID

WORLD_SIZE = 2
WORKER_PHYSICAL_INDICES = (0, 1)
WORKER_GPU_UUID_ENVS = ("H1_WORKER0_GPU_UUID", "H1_WORKER1_GPU_UUID")
WORKER_VICTIM_PORTS = (8000, 8001)
H20_GPU_UUID_ENV = "H1_H20_GPU_UUID"

# Compatibility names used by the superseded exclusive-role implementation.
VICTIM_PHYSICAL_INDEX = 0
ATTACKER_PHYSICAL_INDEX = 1
VICTIM_GPU_UUID_ENV = "H1_VICTIM_GPU_UUID"
ATTACKER_GPU_UUID_ENV = "H1_ATTACKER_GPU_UUID"

VICTIM_DTYPE = "float16"
VICTIM_QUANTIZATION = "none"
ATTACKER_DTYPE = "float16"
ATTACKER_QUANTIZATION = "bitsandbytes-nf4-double-quant-float16"
ATTACKER_BNB_QUANT_TYPE = "nf4"
ATTACKER_BNB_DOUBLE_QUANT = True
ATTACKER_BNB_COMPUTE_DTYPE = "float16"

SOFTWARE_VERSION_DISTRIBUTIONS = (
    "torch", "transformers", "peft", "bitsandbytes", "trl", "accelerate",
)
_GPU_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9][A-Za-z0-9-]{7,127}$")
_FORBIDDEN_PREIMPORT_MODULES = ("torch", "transformers", "peft", "bitsandbytes")
_MINIMUM_V100_32GB_MEMORY_MIB = 30_000
_PREPARED_PRODUCTION_WORKERS: dict[int, dict] = {}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_H20_LEGACY_PROFILE = {
    "schema_version": RUNTIME_PROFILE_SCHEMA_VERSION,
    "profile_id": LEGACY_H20_PROFILE_ID,
    "topology": "single-h20-shared",
    "hardware": {"gpu_count": 1, "gpu_name_contains": "H20"},
    "victim": {"dtype": "bfloat16", "quantization": "fp8", "port": 8000},
    "attacker": {
        "dtype": "bfloat16",
        "quantization": "bitsandbytes-nf4-double-quant-bfloat16",
    },
    "precision_policy": {
        "victim_canonical": "fp8",
        "cross_precision_equivalence_required": False,
        "lifecycle_repeatability": "two-distinct-fp8-processes",
    },
}

_V100_DDP_PROFILE = {
    "schema_version": RUNTIME_PROFILE_SCHEMA_VERSION,
    "profile_id": V100_DDP_PROFILE_ID,
    "topology": "two-colocated-victim-replica-plus-attacker-rank-workers",
    "hardware": {
        "gpu_count": WORLD_SIZE,
        "gpu_name_contains": "V100-SXM2-32GB",
        "compute_capability": [7, 0],
        "minimum_memory_total_mib": _MINIMUM_V100_32GB_MEMORY_MIB,
    },
    "distributed": {
        "world_size": WORLD_SIZE,
        "backend": "nccl",
        "logical_device_index_per_process": 0,
        "worker_mapping": "local_rank_equals_physical_index",
    },
    "workers": [
        {
            "local_rank": rank,
            "physical_index": rank,
            "uuid_env": WORKER_GPU_UUID_ENVS[rank],
            "victim": {
                "port": WORKER_VICTIM_PORTS[rank],
                "dtype": VICTIM_DTYPE,
                "quantization": VICTIM_QUANTIZATION,
            },
            "attacker": {
                "dtype": ATTACKER_DTYPE,
                "quantization": ATTACKER_QUANTIZATION,
                "bnb_4bit_quant_type": ATTACKER_BNB_QUANT_TYPE,
                "bnb_4bit_use_double_quant": ATTACKER_BNB_DOUBLE_QUANT,
                "bnb_4bit_compute_dtype": ATTACKER_BNB_COMPUTE_DTYPE,
            },
        }
        for rank in range(WORLD_SIZE)
    ],
    "isolation": {
        "cuda_device_order": "PCI_BUS_ID",
        "visible_device_selector": "uuid",
        "logical_device_count": 1,
    },
}

_RUNTIME_PROFILES = {
    LEGACY_H20_PROFILE_ID: _H20_LEGACY_PROFILE,
    V100_DDP_PROFILE_ID: _V100_DDP_PROFILE,
}
H20_RUNTIME_PROFILE_SHA256 = canonical_sha256(_H20_LEGACY_PROFILE)
RUNTIME_PROFILE_SHA256 = canonical_sha256(_V100_DDP_PROFILE)


def runtime_profile(profile_id: str = RUNTIME_PROFILE_ID) -> dict:
    try:
        return copy.deepcopy(_RUNTIME_PROFILES[profile_id])
    except KeyError as exc:
        raise ValueError(f"unknown runtime profile: {profile_id!r}") from exc


def legacy_h20_runtime_profile() -> dict:
    return runtime_profile(LEGACY_H20_PROFILE_ID)


def runtime_profile_sha256(profile_id: str = RUNTIME_PROFILE_ID) -> str:
    return canonical_sha256(runtime_profile(profile_id))


def installed_software_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in SOFTWARE_VERSION_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def validate_software_versions(record: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(record, Mapping) or set(record) != set(SOFTWARE_VERSION_DISTRIBUTIONS):
        raise ValueError("software_versions must contain the exact frozen distribution key set")
    result = {str(key): str(value) for key, value in record.items()}
    if any(not value.strip() for value in result.values()):
        raise ValueError("software_versions values must be non-empty")
    return result


def _normalize_inventory_row(raw: Mapping[str, Any]) -> dict:
    try:
        row = {
            "index": int(raw["index"]),
            "uuid": str(raw["uuid"]).strip(),
            "name": str(raw["name"]).strip(),
            "memory_total_mib": int(raw["memory_total_mib"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed GPU inventory row: {raw!r}") from exc
    if (row["index"] < 0 or not _GPU_UUID_RE.fullmatch(row["uuid"])
            or not row["name"] or row["memory_total_mib"] < 1):
        raise ValueError(f"invalid GPU inventory row: {raw!r}")
    return row


def query_nvidia_smi_inventory() -> list[dict]:
    command = [
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"cannot query GPU inventory with nvidia-smi: {exc}") from exc
    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4:
            raise RuntimeError(f"unexpected nvidia-smi inventory line: {line!r}")
        rows.append(_normalize_inventory_row({
            "index": fields[0], "uuid": fields[1], "name": fields[2],
            "memory_total_mib": fields[3],
        }))
    return rows


def _exact_uuid(value: Any, label: str) -> str:
    value = str(value or "").strip()
    if not _GPU_UUID_RE.fullmatch(value):
        raise ValueError(f"{label} must be an exact NVIDIA GPU UUID, got {value!r}")
    return value


def validate_worker_inventory(rows: Sequence[Mapping[str, Any]], worker_uuids: Sequence[str]) -> dict:
    if len(worker_uuids) != WORLD_SIZE:
        raise ValueError(f"expected {WORLD_SIZE} worker UUIDs")
    uuids = [_exact_uuid(value, WORKER_GPU_UUID_ENVS[i]) for i, value in enumerate(worker_uuids)]
    if len(set(uuids)) != WORLD_SIZE:
        raise ValueError("worker GPU UUIDs must be distinct")
    normalized = sorted((_normalize_inventory_row(row) for row in rows), key=lambda row: row["index"])
    if len(normalized) != WORLD_SIZE or [row["index"] for row in normalized] != [0, 1]:
        raise RuntimeError(f"V100 DDP requires exact physical indices [0,1], got {normalized}")
    for rank, row in enumerate(normalized):
        if "v100-sxm2-32gb" not in row["name"].casefold():
            raise RuntimeError(f"worker {rank} is not V100-SXM2-32GB: {row['name']!r}")
        if row["memory_total_mib"] < _MINIMUM_V100_32GB_MEMORY_MIB:
            raise RuntimeError(f"worker {rank} has insufficient memory: {row['memory_total_mib']} MiB")
        if row["uuid"] != uuids[rank]:
            raise RuntimeError(
                f"worker {rank} UUID/index mismatch: env={uuids[rank]}, inventory={row['uuid']}"
            )
    return {
        "profile_id": V100_DDP_PROFILE_ID,
        "profile_sha256": RUNTIME_PROFILE_SHA256,
        "inventory": normalized,
        "workers": [
            {
                **normalized[rank],
                "physical_index": rank,
                "local_rank": rank,
                "victim_port": WORKER_VICTIM_PORTS[rank],
            }
            for rank in range(WORLD_SIZE)
        ],
    }


# Compatibility alias for callers of the superseded exclusive-role validator.
def validate_gpu_inventory(
    rows: Sequence[Mapping[str, Any]], *, victim_uuid: str, attacker_uuid: str,
) -> dict:
    checked = validate_worker_inventory(rows, [victim_uuid, attacker_uuid])
    return {
        "profile_id": checked["profile_id"],
        "profile_sha256": checked["profile_sha256"],
        "inventory": checked["inventory"],
        "bindings": {
            "victim": copy.deepcopy(checked["workers"][0]),
            "attacker": copy.deepcopy(checked["workers"][1]),
        },
    }


def _require_distributed_env(env: Mapping[str, str], local_rank: int) -> tuple[int, int]:
    try:
        rank = int(env.get("RANK", ""))
        world_size = int(env.get("WORLD_SIZE", ""))
        local_world_size = int(env.get("LOCAL_WORLD_SIZE", ""))
        env_local_rank = int(env.get("LOCAL_RANK", ""))
    except ValueError as exc:
        raise RuntimeError("torchrun rank environment is missing or malformed") from exc
    if (world_size, local_world_size) != (WORLD_SIZE, WORLD_SIZE):
        raise RuntimeError(
            f"formal V100 DDP requires WORLD_SIZE=LOCAL_WORLD_SIZE=2, got "
            f"{world_size}/{local_world_size}"
        )
    if local_rank not in (0, 1) or env_local_rank != local_rank or rank != local_rank:
        raise RuntimeError(
            f"single-node rank mapping mismatch: rank={rank}, local_rank={env_local_rank}, "
            f"requested={local_rank}"
        )
    return rank, world_size


def prepare_worker_before_torch(
    local_rank: int | None = None,
    *,
    environ: MutableMapping[str, str] | None = None,
    inventory: Sequence[Mapping[str, Any]] | None = None,
) -> dict:
    """Bind one torchrun worker to its exact UUID before any torch-related import."""
    env = os.environ if environ is None else environ
    if local_rank is None:
        try:
            local_rank = int(env.get("LOCAL_RANK", ""))
        except ValueError as exc:
            raise RuntimeError("LOCAL_RANK must be set by torchrun") from exc
    rank, world_size = _require_distributed_env(env, local_rank)
    production = environ is None and inventory is None
    if production and local_rank in _PREPARED_PRODUCTION_WORKERS:
        prepared = _PREPARED_PRODUCTION_WORKERS[local_rank]
        if env.get("CUDA_VISIBLE_DEVICES") != prepared["binding"]["uuid"]:
            raise RuntimeError("CUDA visibility changed after worker preparation")
        return copy.deepcopy(prepared)
    imported = [name for name in _FORBIDDEN_PREIMPORT_MODULES if name in sys.modules]
    if imported:
        raise RuntimeError(f"worker GPU isolation must precede ML imports; already imported: {imported}")
    worker_uuids = [_exact_uuid(env.get(name, ""), name) for name in WORKER_GPU_UUID_ENVS]
    rows = query_nvidia_smi_inventory() if inventory is None else list(inventory)
    checked = validate_worker_inventory(rows, worker_uuids)
    binding = checked["workers"][local_rank]
    current_order = env.get("CUDA_DEVICE_ORDER")
    if current_order not in (None, "", "PCI_BUS_ID"):
        raise RuntimeError(f"conflicting CUDA_DEVICE_ORDER={current_order!r}")
    current_visible = env.get("CUDA_VISIBLE_DEVICES")
    if current_visible not in (None, "", binding["uuid"]):
        raise RuntimeError(
            f"conflicting CUDA_VISIBLE_DEVICES={current_visible!r}; expected {binding['uuid']}"
        )
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = binding["uuid"]
    prepared = {
        "schema_version": RUNTIME_PROFILE_SCHEMA_VERSION,
        "kind": "h1_runtime_pre_torch_binding",
        "profile_id": V100_DDP_PROFILE_ID,
        "profile": runtime_profile(),
        "profile_sha256": RUNTIME_PROFILE_SHA256,
        "role": "worker",
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "binding": copy.deepcopy(binding),
        "inventory": copy.deepcopy(checked["inventory"]),
        "victim_endpoint": f"http://127.0.0.1:{binding['victim_port']}/v1",
        "environment": {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": binding["uuid"],
        },
    }
    if production:
        _PREPARED_PRODUCTION_WORKERS[local_rank] = copy.deepcopy(prepared)
    return prepared


def prepare_role_before_torch(
    role: str, *, environ: MutableMapping[str, str] | None = None,
    inventory: Sequence[Mapping[str, Any]] | None = None,
) -> dict:
    """Removed exclusive-role fallback; kept only as an explicit fail-closed API tombstone."""
    del role, environ, inventory
    raise RuntimeError(
        "exclusive victim/attacker V100 role binding is forbidden; use "
        "prepare_worker_before_torch(local_rank). H20 uses its separate legacy profile."
    )


def validate_torch_runtime(role: str, torch_module: Any, pre_torch_record: Mapping[str, Any]) -> dict:
    """Verify one visible logical Volta device; ignore PyTorch's unreliable BF16 helper."""
    if role not in ("worker", "victim", "attacker") or pre_torch_record.get("role") != role:
        raise ValueError(f"runtime role mismatch: expected {role!r}")
    cuda = torch_module.cuda
    if not cuda.is_available() or int(cuda.device_count()) != 1:
        raise RuntimeError("each worker must see exactly one CUDA device")
    name = str(cuda.get_device_name(0))
    capability = tuple(int(value) for value in cuda.get_device_capability(0))
    if "v100" not in name.casefold() or capability != (7, 0):
        raise RuntimeError(f"logical cuda:0 is not V100 Volta: {name!r}/{capability}")
    return {
        "logical_device_index": 0,
        "logical_device_count": 1,
        "device_name": name,
        "compute_capability": [7, 0],
        "torch_version": str(torch_module.__version__),
        "cuda_version": str(torch_module.version.cuda),
        "selected_uuid": pre_torch_record["binding"]["uuid"],
    }


def runtime_manifest(
    pre_torch_record: Mapping[str, Any], post_torch_record: Mapping[str, Any] | None = None,
) -> dict:
    document = copy.deepcopy(dict(pre_torch_record))
    document["kind"] = "h1_runtime_manifest"
    document["torch"] = copy.deepcopy(dict(post_torch_record)) if post_torch_record else None
    return validate_runtime_manifest(document, require_role=document.get("role"))


def validate_runtime_manifest(
    record: Mapping[str, Any], *, require_role: str | None = None, allow_pre_torch: bool = False,
) -> dict:
    if not isinstance(record, Mapping):
        raise ValueError("runtime manifest must be a mapping")
    expected_kind = "h1_runtime_pre_torch_binding" if allow_pre_torch else "h1_runtime_manifest"
    mismatches: dict[str, Any] = {}
    if record.get("schema_version") != RUNTIME_PROFILE_SCHEMA_VERSION:
        mismatches["schema_version"] = record.get("schema_version")
    if record.get("kind") != expected_kind:
        mismatches["kind"] = record.get("kind")
    if record.get("profile_id") != V100_DDP_PROFILE_ID or record.get("profile") != _V100_DDP_PROFILE:
        mismatches["profile"] = "not the canonical co-located V100 DDP profile"
    if record.get("profile_sha256") != RUNTIME_PROFILE_SHA256:
        mismatches["profile_sha256"] = record.get("profile_sha256")
    role = record.get("role")
    if role != "worker" or (require_role is not None and require_role != "worker"):
        mismatches["role"] = role
    local_rank = record.get("local_rank")
    if local_rank not in (0, 1) or record.get("rank") != local_rank or record.get("world_size") != 2:
        mismatches["distributed"] = {
            "rank": record.get("rank"), "local_rank": local_rank,
            "world_size": record.get("world_size"),
        }
    inventory = record.get("inventory")
    binding = record.get("binding")
    canonical_binding = None
    try:
        normalized = sorted((_normalize_inventory_row(row) for row in inventory), key=lambda row: row["index"])
        checked = validate_worker_inventory(normalized, [row["uuid"] for row in normalized])
        if list(inventory) != checked["inventory"]:
            raise ValueError("inventory is not canonical/sorted")
        if local_rank in (0, 1):
            canonical_binding = checked["workers"][local_rank]
            if dict(binding or {}) != canonical_binding:
                raise ValueError("binding does not exactly equal the inventory row/rank/port")
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        mismatches["inventory_binding"] = str(exc)
    expected_endpoint = (
        f"http://127.0.0.1:{WORKER_VICTIM_PORTS[local_rank]}/v1"
        if local_rank in (0, 1) else None
    )
    if record.get("victim_endpoint") != expected_endpoint:
        mismatches["victim_endpoint"] = record.get("victim_endpoint")
    environment = record.get("environment") or {}
    if (not isinstance(binding, Mapping)
            or environment.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            or environment.get("CUDA_VISIBLE_DEVICES") != binding.get("uuid")):
        mismatches["environment"] = environment
    if not allow_pre_torch:
        torch_record = record.get("torch")
        if (not isinstance(torch_record, Mapping)
                or torch_record.get("logical_device_index") != 0
                or torch_record.get("logical_device_count") != 1
                or torch_record.get("compute_capability") != [7, 0]
                or torch_record.get("selected_uuid") != (binding or {}).get("uuid")
                or (canonical_binding and torch_record.get("device_name") != canonical_binding["name"])):
            mismatches["torch"] = torch_record
    if mismatches:
        raise ValueError(f"runtime manifest mismatch: {mismatches}")
    return copy.deepcopy(dict(record))
