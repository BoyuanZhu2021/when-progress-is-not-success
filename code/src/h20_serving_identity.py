"""Pure, sealed identity contract for the single-H20 vLLM victim.

The serving launcher, FP8 lifecycle proof, formal Gate, trainer, and
evaluation all consume this module.  It deliberately performs no process,
network, CUDA, or filesystem I/O: callers must obtain live observations and
then validate them here.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from .model_pins import (
    REMOTE_HF_HOME,
    VICTIM_H20_SERVED_NAME,
    VICTIM_HF_MODEL,
    VICTIM_REVISION,
)
from .runtime_profile import (
    H20_GPU_UUID_ENV,
    H20_RUNTIME_PROFILE_SHA256,
    LEGACY_H20_PROFILE_ID,
    canonical_sha256,
)
from .victim_decision_protocol import STRUCTURED_OUTPUT_CONFIG_JSON


import os
TMP = os.environ.get("OFC_TMP", "/tmp")
HOST = "127.0.0.1"
PORT = 8000
ENDPOINT = f"http://{HOST}:{PORT}/v1"
VLLM_ENV = "/root/miniconda3/envs/vllm"
VLLM_PYTHON = f"{VLLM_ENV}/bin/python"
VLLM_BIN = f"{VLLM_ENV}/bin/vllm"
VLLM_VERSION = "0.24.0"
CUDA_HOME = "/usr/local/cuda"
VLLM_NINJA = f"{VLLM_ENV}/bin/ninja"
VLLM_NINJA_METADATA_VERSION = "1.13.0"
VLLM_NINJA_BINARY_VERSION = "1.13.0.git.kitware.jobserver-pipe-1"
CUDA_NVCC = f"{CUDA_HOME}/bin/nvcc"
CUDA_NVCC_RELEASE = "12.8"
SYSTEM_CXX = "/usr/bin/c++"
VLLM_TOOLCHAIN_PATH = ":".join((
    f"{VLLM_ENV}/bin",
    f"{CUDA_HOME}/bin",
    "/usr/local/sbin",
    "/usr/local/bin",
    "/usr/sbin",
    "/usr/bin",
    "/sbin",
    "/bin",
))
MANIFEST_PATH = f"{TMP}/h1_victim_manifest.json"
MANIFEST_KIND = "h1_h20_victim_service_manifest"
RUNTIME_REFERENCE_KIND = "h1_h20_restored_fp8_runtime_reference"
FORMAL_RUNTIME_BUNDLE_KIND = "h1_h20_formal_runtime_bundle"
LIVE_CHECK_KIND = "h1_h20_live_runtime_check"
RUNTIME_EQUIVALENCE_KIND = "h1_h20_runtime_protocol_equivalence"
SCHEMA_VERSION = 1
SUPPORTED_QUANTIZATIONS = ("fp8", "bf16")

MODEL_IDENTITY = {
    "hf_model": VICTIM_HF_MODEL,
    "revision": VICTIM_REVISION,
    "served_model": VICTIM_H20_SERVED_NAME,
}

_STATIC_ENV = {
    "HF_HOME": REMOTE_HF_HOME,
    "HF_HUB_DISABLE_XET": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "OMP_NUM_THREADS": "16",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER": "0",
    "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
    "H1_RUNTIME_PROFILE_ID": LEGACY_H20_PROFILE_ID,
    "H1_RUNTIME_PROFILE_SHA256": H20_RUNTIME_PROFILE_SHA256,
    "H1_VICTIM_BACKEND": "vllm",
    "H1_VICTIM_HF_MODEL": VICTIM_HF_MODEL,
    "H1_VICTIM_REVISION": VICTIM_REVISION,
    "H1_VICTIM_SERVED_MODEL": VICTIM_H20_SERVED_NAME,
    "H1_VICTIM_DTYPE": "bfloat16",
    "H1_VICTIM_HOST": HOST,
    "H1_VICTIM_PORT": str(PORT),
}
IDENTITY_ENV_KEYS = tuple(sorted({
    "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES",
    H20_GPU_UUID_ENV,
    "H1_VICTIM_QUANTIZATION",
    *_STATIC_ENV.keys(),
}))

_MANIFEST_KEYS = {
    "schema_version", "kind", "path", "profile_id", "profile_sha256",
    "model", "backend", "endpoint", "gpu", "process", "service",
    "sealed_at", "payload_sha256",
}
_REFERENCE_KEYS = {
    "schema_version", "kind", "profile_id", "profile_sha256",
    "service_manifest_path", "service_manifest_payload_sha256", "model",
    "backend", "endpoint", "gpu_uuid", "process", "sealed_at",
    "payload_sha256",
}
_BUNDLE_KEYS = {
    "schema_version", "kind", "profile_id", "profile_sha256",
    "quant_cycle_status_payload_sha256", "restored_fp8_runtime",
    "gate_checks", "sealed_at", "payload_sha256",
}
_LIVE_CHECK_KEYS = {
    "schema_version", "kind", "phase", "verified_at",
    "runtime_reference_payload_sha256", "service_manifest_payload_sha256",
    "gpu_uuid", "process", "payload_sha256",
}


def seal(document: Mapping[str, Any]) -> dict:
    """Return a deep-copied canonical JSON seal."""
    payload = copy.deepcopy(dict(document))
    payload.pop("payload_sha256", None)
    payload["payload_sha256"] = canonical_sha256(payload)
    return payload


def validate_seal(document: Mapping[str, Any]) -> dict:
    if not isinstance(document, Mapping):
        raise ValueError("sealed document must be a mapping")
    result = copy.deepcopy(dict(document))
    claimed = result.pop("payload_sha256", None)
    expected = canonical_sha256(result)
    if claimed != expected:
        raise ValueError(f"H20 identity seal mismatch: expected {expected}, got {claimed}")
    result["payload_sha256"] = claimed
    return result


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def backend_identity(quantization: str) -> dict:
    if quantization not in SUPPORTED_QUANTIZATIONS:
        raise ValueError(f"unsupported H20 victim quantization: {quantization!r}")
    return {
        "name": "vllm",
        "version": VLLM_VERSION,
        "dtype": "bfloat16",
        "quantization": quantization,
    }


def expected_cmdline(quantization: str) -> list[str]:
    """Canonical ordered argv for both the launcher and quantization cycle."""
    command = [
        VLLM_PYTHON,
        VLLM_BIN,
        "serve",
        VICTIM_HF_MODEL,
        "--revision",
        VICTIM_REVISION,
        "--served-model-name",
        VICTIM_H20_SERVED_NAME,
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        "0.55",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "256",
        "--structured-outputs-config",
        STRUCTURED_OUTPUT_CONFIG_JSON,
    ]
    if quantization == "fp8":
        command.extend(["--quantization", "fp8"])
    elif quantization != "bf16":
        raise ValueError(f"unsupported H20 victim quantization: {quantization!r}")
    return command


def expected_environment(quantization: str, gpu_uuid: str) -> dict[str, str]:
    backend_identity(quantization)
    if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
        raise ValueError(f"invalid H20 GPU UUID: {gpu_uuid!r}")
    return dict(sorted({
        **_STATIC_ENV,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": gpu_uuid,
        H20_GPU_UUID_ENV: gpu_uuid,
        "H1_VICTIM_QUANTIZATION": quantization,
    }.items()))


def expected_toolchain_environment() -> dict[str, str]:
    """Exact child-only JIT toolchain environment.

    These keys intentionally remain outside ``IDENTITY_ENV_KEYS`` so a sealed
    stale manifest from the pre-repair dead lifecycle can still be validated
    and removed.  The new lifecycle's full raw environment SHA-256 still binds
    their bytes after launch.
    """
    return {"CUDA_HOME": CUDA_HOME, "PATH": VLLM_TOOLCHAIN_PATH}


def build_service_manifest(
    *,
    quantization: str,
    gpu: Mapping[str, Any],
    process: Mapping[str, Any],
    service: Mapping[str, Any],
    sealed_at: str,
) -> dict:
    """Build and validate the only accepted H20 service-manifest schema."""
    document = seal({
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "path": MANIFEST_PATH,
        "profile_id": LEGACY_H20_PROFILE_ID,
        "profile_sha256": H20_RUNTIME_PROFILE_SHA256,
        "model": MODEL_IDENTITY,
        "backend": backend_identity(quantization),
        "endpoint": ENDPOINT,
        "gpu": copy.deepcopy(dict(gpu)),
        "process": copy.deepcopy(dict(process)),
        "service": copy.deepcopy(dict(service)),
        "sealed_at": sealed_at,
    })
    return validate_service_manifest(document, expected_quantization=quantization)


def validate_service_manifest(
    document: Mapping[str, Any], *, expected_quantization: str | None = None,
) -> dict:
    result = validate_seal(document)
    if set(result) != _MANIFEST_KEYS:
        raise ValueError("H20 service manifest field set mismatch")
    if result.get("schema_version") != SCHEMA_VERSION or result.get("kind") != MANIFEST_KIND:
        raise ValueError("unexpected H20 service manifest schema/kind")
    if result.get("path") != MANIFEST_PATH:
        raise ValueError("H20 service manifest path mismatch")
    if (result.get("profile_id") != LEGACY_H20_PROFILE_ID
            or result.get("profile_sha256") != H20_RUNTIME_PROFILE_SHA256):
        raise ValueError("H20 service manifest runtime profile mismatch")
    if result.get("model") != MODEL_IDENTITY:
        raise ValueError("H20 victim model identity mismatch")
    backend = result.get("backend")
    quantization = backend.get("quantization") if isinstance(backend, Mapping) else None
    if expected_quantization is not None and quantization != expected_quantization:
        raise ValueError("H20 victim quantization mismatch")
    if backend != backend_identity(quantization):
        raise ValueError("H20 victim backend identity mismatch")
    if result.get("endpoint") != ENDPOINT:
        raise ValueError("H20 victim endpoint mismatch")

    gpu = result.get("gpu")
    if not isinstance(gpu, Mapping) or set(gpu) != {
        "index", "uuid", "name", "memory_total_mib"
    }:
        raise ValueError("H20 GPU inventory field set mismatch")
    if (gpu.get("index") != 0
            or not isinstance(gpu.get("uuid"), str)
            or not gpu["uuid"].startswith("GPU-")
            or not isinstance(gpu.get("name"), str)
            or "H20" not in gpu["name"].upper()
            or not isinstance(gpu.get("memory_total_mib"), int)
            or isinstance(gpu.get("memory_total_mib"), bool)
            or gpu["memory_total_mib"] < 90_000):
        raise ValueError("service manifest does not identify the exact single H20")

    process = result.get("process")
    if not isinstance(process, Mapping) or set(process) != {
        "pid", "start_time_ticks", "cmdline", "cmdline_sha256",
        "environ_sha256", "environment", "gpu_uuid",
    }:
        raise ValueError("H20 process identity field set mismatch")
    if (not isinstance(process.get("pid"), int)
            or isinstance(process.get("pid"), bool)
            or process["pid"] < 2
            or not isinstance(process.get("start_time_ticks"), int)
            or isinstance(process.get("start_time_ticks"), bool)
            or process["start_time_ticks"] < 1):
        raise ValueError("invalid H20 process PID/start ticks")
    command = expected_cmdline(quantization)
    if process.get("cmdline") != command:
        raise ValueError("H20 process cmdline is not canonical ordered argv")
    if process.get("cmdline_sha256") != canonical_sha256(command):
        raise ValueError("H20 process cmdline hash mismatch")
    if process.get("environment") != expected_environment(quantization, gpu["uuid"]):
        raise ValueError("H20 process identity environment is not canonical/exact")
    if not _is_sha256(process.get("environ_sha256")):
        raise ValueError("H20 process full environment hash is malformed")
    if process.get("gpu_uuid") != gpu["uuid"]:
        raise ValueError("H20 process/GPU UUID mismatch")

    service = result.get("service")
    if (not isinstance(service, Mapping)
            or set(service) != {"endpoint", "model_ids", "observed_at"}
            or service.get("endpoint") != ENDPOINT
            or service.get("model_ids") != [VICTIM_H20_SERVED_NAME]
            or not isinstance(service.get("observed_at"), str)
            or not service["observed_at"]):
        raise ValueError("H20 service API identity mismatch")
    if not isinstance(result.get("sealed_at"), str) or not result["sealed_at"]:
        raise ValueError("H20 service manifest sealed_at missing")
    return result


def runtime_reference(service_manifest: Mapping[str, Any]) -> dict:
    """Seal the exact restored process identity embedded in quant/Gate artifacts."""
    manifest = validate_service_manifest(service_manifest, expected_quantization="fp8")
    process = manifest["process"]
    reference = seal({
        "schema_version": SCHEMA_VERSION,
        "kind": RUNTIME_REFERENCE_KIND,
        "profile_id": LEGACY_H20_PROFILE_ID,
        "profile_sha256": H20_RUNTIME_PROFILE_SHA256,
        "service_manifest_path": MANIFEST_PATH,
        "service_manifest_payload_sha256": manifest["payload_sha256"],
        "model": manifest["model"],
        "backend": manifest["backend"],
        "endpoint": manifest["endpoint"],
        "gpu_uuid": manifest["gpu"]["uuid"],
        "process": {
            "pid": process["pid"],
            "start_time_ticks": process["start_time_ticks"],
            "cmdline_sha256": process["cmdline_sha256"],
            "environ_sha256": process["environ_sha256"],
        },
        "sealed_at": manifest["sealed_at"],
    })
    return validate_runtime_reference(reference, manifest)


def validate_runtime_reference(
    reference: Mapping[str, Any], service_manifest: Mapping[str, Any] | None = None,
) -> dict:
    result = validate_seal(reference)
    if set(result) != _REFERENCE_KEYS:
        raise ValueError("H20 runtime reference field set mismatch")
    if (result.get("schema_version") != SCHEMA_VERSION
            or result.get("kind") != RUNTIME_REFERENCE_KIND
            or result.get("profile_id") != LEGACY_H20_PROFILE_ID
            or result.get("profile_sha256") != H20_RUNTIME_PROFILE_SHA256):
        raise ValueError("H20 runtime reference schema/profile mismatch")
    if (result.get("service_manifest_path") != MANIFEST_PATH
            or not _is_sha256(result.get("service_manifest_payload_sha256"))
            or result.get("model") != MODEL_IDENTITY
            or result.get("backend") != backend_identity("fp8")
            or result.get("endpoint") != ENDPOINT
            or not isinstance(result.get("gpu_uuid"), str)
            or not result["gpu_uuid"].startswith("GPU-")):
        raise ValueError("H20 runtime reference identity mismatch")
    process = result.get("process")
    if (not isinstance(process, Mapping)
            or set(process) != {
                "pid", "start_time_ticks", "cmdline_sha256", "environ_sha256"
            }
            or not isinstance(process.get("pid"), int)
            or process["pid"] < 2
            or not isinstance(process.get("start_time_ticks"), int)
            or process["start_time_ticks"] < 1
            or not _is_sha256(process.get("cmdline_sha256"))
            or not _is_sha256(process.get("environ_sha256"))):
        raise ValueError("H20 runtime reference process mismatch")
    if not isinstance(result.get("sealed_at"), str) or not result["sealed_at"]:
        raise ValueError("H20 runtime reference sealed_at missing")
    if service_manifest is not None:
        manifest = validate_service_manifest(service_manifest, expected_quantization="fp8")
        process_manifest = manifest["process"]
        direct = {
            "service_manifest_payload_sha256": manifest["payload_sha256"],
            "model": manifest["model"],
            "backend": manifest["backend"],
            "endpoint": manifest["endpoint"],
            "gpu_uuid": manifest["gpu"]["uuid"],
            "process": {
                "pid": process_manifest["pid"],
                "start_time_ticks": process_manifest["start_time_ticks"],
                "cmdline_sha256": process_manifest["cmdline_sha256"],
                "environ_sha256": process_manifest["environ_sha256"],
            },
            "sealed_at": manifest["sealed_at"],
        }
        if any(result.get(key) != value for key, value in direct.items()):
            raise ValueError("restored FP8 runtime reference differs from live service manifest")
    return result


def live_runtime_check(
    reference: Mapping[str, Any], service_manifest: Mapping[str, Any], *, phase: str,
    verified_at: str,
) -> dict:
    checked = validate_runtime_reference(reference, service_manifest)
    manifest = validate_service_manifest(service_manifest, expected_quantization="fp8")
    document = seal({
        "schema_version": SCHEMA_VERSION,
        "kind": LIVE_CHECK_KIND,
        "phase": phase,
        "verified_at": verified_at,
        "runtime_reference_payload_sha256": checked["payload_sha256"],
        "service_manifest_payload_sha256": manifest["payload_sha256"],
        "gpu_uuid": manifest["gpu"]["uuid"],
        "process": copy.deepcopy(checked["process"]),
    })
    return validate_live_runtime_check(document, checked, expected_phase=phase)


def validate_live_runtime_check(
    check: Mapping[str, Any], reference: Mapping[str, Any], *, expected_phase: str | None = None,
) -> dict:
    result = validate_seal(check)
    runtime = validate_runtime_reference(reference)
    if set(result) != _LIVE_CHECK_KEYS:
        raise ValueError("H20 live runtime check field set mismatch")
    if result.get("schema_version") != SCHEMA_VERSION or result.get("kind") != LIVE_CHECK_KIND:
        raise ValueError("H20 live runtime check schema/kind mismatch")
    if expected_phase is not None and result.get("phase") != expected_phase:
        raise ValueError("H20 live runtime check phase mismatch")
    if not isinstance(result.get("phase"), str) or not result["phase"]:
        raise ValueError("H20 live runtime check phase missing")
    if not isinstance(result.get("verified_at"), str) or not result["verified_at"]:
        raise ValueError("H20 live runtime check timestamp missing")
    expected = {
        "runtime_reference_payload_sha256": runtime["payload_sha256"],
        "service_manifest_payload_sha256": runtime["service_manifest_payload_sha256"],
        "gpu_uuid": runtime["gpu_uuid"],
        "process": runtime["process"],
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("H20 live runtime check differs from restored FP8 reference")
    return result


def build_formal_runtime_bundle(
    *,
    quant_cycle_status_payload_sha256: str,
    restored_fp8_runtime: Mapping[str, Any],
    gate_checks: Mapping[str, Any] | None = None,
    sealed_at: str,
) -> dict:
    if not _is_sha256(quant_cycle_status_payload_sha256):
        raise ValueError("quant cycle status payload hash malformed")
    reference = validate_runtime_reference(restored_fp8_runtime)
    document = seal({
        "schema_version": SCHEMA_VERSION,
        "kind": FORMAL_RUNTIME_BUNDLE_KIND,
        "profile_id": LEGACY_H20_PROFILE_ID,
        "profile_sha256": H20_RUNTIME_PROFILE_SHA256,
        "quant_cycle_status_payload_sha256": quant_cycle_status_payload_sha256,
        "restored_fp8_runtime": reference,
        "gate_checks": copy.deepcopy(dict(gate_checks or {})),
        "sealed_at": sealed_at,
    })
    return validate_h20_formal_runtime_bundle(document, require_gate_checks=False)


def with_gate_runtime_check(bundle: Mapping[str, Any], check: Mapping[str, Any]) -> dict:
    current = validate_h20_formal_runtime_bundle(bundle, require_gate_checks=False)
    phase = check.get("phase") if isinstance(check, Mapping) else None
    if phase not in {"gate_open", "gate_close"}:
        raise ValueError(f"unsupported Gate runtime phase: {phase!r}")
    verified = validate_live_runtime_check(
        check, current["restored_fp8_runtime"], expected_phase=phase
    )
    checks = copy.deepcopy(current["gate_checks"])
    if phase in checks:
        raise ValueError(f"Gate runtime phase already recorded: {phase}")
    checks[phase] = verified
    current["gate_checks"] = checks
    return seal(current)


def validate_h20_formal_runtime_bundle(
    bundle: Mapping[str, Any],
    service_manifest: Mapping[str, Any] | None = None,
    *,
    require_gate_checks: bool = True,
) -> dict:
    """Validate the portable lifecycle-proof/Gate runtime contract.

    When ``service_manifest`` is supplied it must be the same currently live
    restored-FP8 process.  The return value is the validated canonical bundle;
    use :func:`canonical_provenance_summary` for a compact run-config record.
    """
    result = validate_seal(bundle)
    if set(result) != _BUNDLE_KEYS:
        raise ValueError("H20 formal runtime bundle field set mismatch")
    if (result.get("schema_version") != SCHEMA_VERSION
            or result.get("kind") != FORMAL_RUNTIME_BUNDLE_KIND
            or result.get("profile_id") != LEGACY_H20_PROFILE_ID
            or result.get("profile_sha256") != H20_RUNTIME_PROFILE_SHA256
            or not _is_sha256(result.get("quant_cycle_status_payload_sha256"))):
        raise ValueError("H20 formal runtime bundle schema/profile mismatch")
    reference = validate_runtime_reference(result.get("restored_fp8_runtime"), service_manifest)
    checks = result.get("gate_checks")
    if not isinstance(checks, Mapping) or set(checks) - {"gate_open", "gate_close"}:
        raise ValueError("H20 formal runtime bundle has unknown Gate checks")
    for phase, check in checks.items():
        validate_live_runtime_check(check, reference, expected_phase=phase)
    if require_gate_checks and set(checks) != {"gate_open", "gate_close"}:
        raise ValueError("H20 formal runtime bundle lacks Gate open/close checks")
    if not isinstance(result.get("sealed_at"), str) or not result["sealed_at"]:
        raise ValueError("H20 formal runtime bundle sealed_at missing")
    return result


def canonical_provenance_summary(bundle: Mapping[str, Any]) -> dict:
    """Compact, canonical record safe to embed in train/eval run configs."""
    checked = validate_h20_formal_runtime_bundle(bundle, require_gate_checks=True)
    runtime = checked["restored_fp8_runtime"]
    return {
        "profile_id": checked["profile_id"],
        "profile_sha256": checked["profile_sha256"],
        "runtime_bundle_payload_sha256": checked["payload_sha256"],
        "quant_cycle_status_payload_sha256": checked[
            "quant_cycle_status_payload_sha256"
        ],
        "runtime_reference_payload_sha256": runtime["payload_sha256"],
        "service_manifest_path": runtime["service_manifest_path"],
        "service_manifest_payload_sha256": runtime[
            "service_manifest_payload_sha256"
        ],
        "model": copy.deepcopy(runtime["model"]),
        "backend": copy.deepcopy(runtime["backend"]),
        "endpoint": runtime["endpoint"],
        "gpu_uuid": runtime["gpu_uuid"],
        "process": copy.deepcopy(runtime["process"]),
        "gate_check_payload_sha256": {
            phase: checked["gate_checks"][phase]["payload_sha256"]
            for phase in ("gate_open", "gate_close")
        },
    }


def validate_h20_runtime_protocol_equivalence(
    immutable_gate_bundle: Mapping[str, Any],
    fresh_runtime_bundle: Mapping[str, Any],
) -> dict:
    """Prove a fresh lifecycle is protocol-equivalent while allowing a new process.

    The calibration/Gate lifecycle may be stopped before final OOD evaluation. A fresh
    FP8-A/FP8-B cycle must therefore produce a new runtime reference. PID, start ticks,
    environment hash, service-manifest hash, and reference hash are intentionally *not* compared;
    profile, model, backend, endpoint, and physical H20 UUID remain immutable.
    """
    gate = validate_h20_formal_runtime_bundle(
        immutable_gate_bundle, require_gate_checks=True
    )
    fresh = validate_h20_formal_runtime_bundle(
        fresh_runtime_bundle, require_gate_checks=False
    )
    gate_runtime = gate["restored_fp8_runtime"]
    fresh_runtime = fresh["restored_fp8_runtime"]
    _validate_distinct_h20_lifecycle(gate, fresh)
    immutable = {
        "profile_id": gate["profile_id"],
        "profile_sha256": gate["profile_sha256"],
        "model": gate_runtime["model"],
        "backend": gate_runtime["backend"],
        "endpoint": gate_runtime["endpoint"],
        "gpu_uuid": gate_runtime["gpu_uuid"],
    }
    fresh_immutable = {
        "profile_id": fresh["profile_id"],
        "profile_sha256": fresh["profile_sha256"],
        "model": fresh_runtime["model"],
        "backend": fresh_runtime["backend"],
        "endpoint": fresh_runtime["endpoint"],
        "gpu_uuid": fresh_runtime["gpu_uuid"],
    }
    if fresh_immutable != immutable:
        raise ValueError(
            "fresh H20 runtime is not protocol-equivalent to immutable Gate runtime"
        )
    document = seal({
        "schema_version": SCHEMA_VERSION,
        "kind": RUNTIME_EQUIVALENCE_KIND,
        "status": "EQUIVALENT_NEW_LIFECYCLE",
        "immutable_gate_runtime_bundle_payload_sha256": gate["payload_sha256"],
        "fresh_runtime_bundle_payload_sha256": fresh["payload_sha256"],
        "immutable_identity": immutable,
        "allowed_fresh_fields": [
            "quant_cycle_status_payload_sha256",
            "restored_fp8_runtime.service_manifest_payload_sha256",
            "restored_fp8_runtime.payload_sha256",
            "restored_fp8_runtime.process.pid",
            "restored_fp8_runtime.process.start_time_ticks",
            "restored_fp8_runtime.process.environ_sha256",
            "restored_fp8_runtime.sealed_at",
            "sealed_at",
        ],
    })
    return validate_h20_runtime_protocol_equivalence_document(
        document, immutable_gate_bundle=gate, fresh_runtime_bundle=fresh
    )


def _validate_distinct_h20_lifecycle(gate: Mapping[str, Any], fresh: Mapping[str, Any]) -> None:
    """Require real new-cycle evidence, not merely a re-labelled Gate bundle.

    The immutable identity intentionally stays constant, but a fresh quantization
    cycle must produce a different restored process and different sealed source
    documents.  Comparing the PID/start-tick tuple permits normal PID reuse while
    rejecting reuse of the same Linux process lifecycle.
    """
    gate_runtime = gate["restored_fp8_runtime"]
    fresh_runtime = fresh["restored_fp8_runtime"]
    if fresh.get("gate_checks"):
        raise ValueError("fresh H20 lifecycle bundle must not reuse Gate checks")
    gate_process = gate_runtime["process"]
    fresh_process = fresh_runtime["process"]
    if (
        gate_process["pid"], gate_process["start_time_ticks"]
    ) == (
        fresh_process["pid"], fresh_process["start_time_ticks"]
    ):
        raise ValueError("fresh H20 runtime reused the immutable Gate process lifecycle")
    distinct_seals = (
        (
            "quant cycle status",
            gate["quant_cycle_status_payload_sha256"],
            fresh["quant_cycle_status_payload_sha256"],
        ),
        (
            "service manifest",
            gate_runtime["service_manifest_payload_sha256"],
            fresh_runtime["service_manifest_payload_sha256"],
        ),
        (
            "runtime reference",
            gate_runtime["payload_sha256"],
            fresh_runtime["payload_sha256"],
        ),
        ("runtime bundle", gate["payload_sha256"], fresh["payload_sha256"]),
    )
    for label, previous, current in distinct_seals:
        if current == previous:
            raise ValueError(f"fresh H20 lifecycle reused the Gate {label} seal")


def validate_h20_runtime_protocol_equivalence_document(
    document: Mapping[str, Any],
    *,
    immutable_gate_bundle: Mapping[str, Any],
    fresh_runtime_bundle: Mapping[str, Any],
) -> dict:
    result = validate_seal(document)
    gate = validate_h20_formal_runtime_bundle(
        immutable_gate_bundle, require_gate_checks=True
    )
    fresh = validate_h20_formal_runtime_bundle(
        fresh_runtime_bundle, require_gate_checks=False
    )
    _validate_distinct_h20_lifecycle(gate, fresh)
    if (result.get("schema_version") != SCHEMA_VERSION
            or result.get("kind") != RUNTIME_EQUIVALENCE_KIND
            or result.get("status") != "EQUIVALENT_NEW_LIFECYCLE"):
        raise ValueError("H20 runtime protocol-equivalence schema/status mismatch")
    if (result.get("immutable_gate_runtime_bundle_payload_sha256")
            != gate["payload_sha256"]
            or result.get("fresh_runtime_bundle_payload_sha256")
            != fresh["payload_sha256"]):
        raise ValueError("H20 runtime protocol-equivalence bundle binding mismatch")
    gate_runtime = gate["restored_fp8_runtime"]
    expected_identity = {
        "profile_id": gate["profile_id"],
        "profile_sha256": gate["profile_sha256"],
        "model": gate_runtime["model"],
        "backend": gate_runtime["backend"],
        "endpoint": gate_runtime["endpoint"],
        "gpu_uuid": gate_runtime["gpu_uuid"],
    }
    if result.get("immutable_identity") != expected_identity:
        raise ValueError("H20 runtime protocol-equivalence identity summary mismatch")
    fresh_runtime = fresh["restored_fp8_runtime"]
    fresh_identity = {
        "profile_id": fresh["profile_id"],
        "profile_sha256": fresh["profile_sha256"],
        "model": fresh_runtime["model"],
        "backend": fresh_runtime["backend"],
        "endpoint": fresh_runtime["endpoint"],
        "gpu_uuid": fresh_runtime["gpu_uuid"],
    }
    if fresh_identity != expected_identity:
        raise ValueError("fresh H20 lifecycle immutable identity mismatch")
    if result.get("allowed_fresh_fields") != [
        "quant_cycle_status_payload_sha256",
        "restored_fp8_runtime.service_manifest_payload_sha256",
        "restored_fp8_runtime.payload_sha256",
        "restored_fp8_runtime.process.pid",
        "restored_fp8_runtime.process.start_time_ticks",
        "restored_fp8_runtime.process.environ_sha256",
        "restored_fp8_runtime.sealed_at",
        "sealed_at",
    ]:
        raise ValueError("H20 runtime protocol-equivalence allowed-field policy mismatch")
    return result


def fresh_quant_artifact_registry(quantization_check: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """Pure explicit registry for comparison/cycle/FP8/BF16 source bytes."""
    fields = (
        ("comparison", "comparison_path", "comparison_file_sha256"),
        ("cycle_status", "cycle_status_path", "cycle_status_file_sha256"),
        ("fp8", "fp8_artifact_path", "fp8_artifact_file_sha256"),
        ("bf16", "bf16_artifact_path", "bf16_artifact_file_sha256"),
    )
    result: list[tuple[str, str, str]] = []
    for label, path_key, hash_key in fields:
        path, digest = quantization_check.get(path_key), quantization_check.get(hash_key)
        if not isinstance(path, str) or not path or not _is_sha256(digest):
            raise ValueError(f"fresh quant artifact registry malformed: {label}")
        result.append((label, path, digest))
    return result


def fresh_fp8_artifact_registry(quantization_check: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """Pure registry for the active FP8-A/FP8-B repeatability evidence."""
    fields = (
        ("repeatability", "repeatability_path", "repeatability_file_sha256"),
        ("cycle_status", "cycle_status_path", "cycle_status_file_sha256"),
        ("fp8_a", "fp8_a_artifact_path", "fp8_a_artifact_file_sha256"),
        ("fp8_b", "fp8_b_artifact_path", "fp8_b_artifact_file_sha256"),
    )
    result: list[tuple[str, str, str]] = []
    for label, path_key, hash_key in fields:
        path, digest = quantization_check.get(path_key), quantization_check.get(hash_key)
        if not isinstance(path, str) or not path or not _is_sha256(digest):
            raise ValueError(f"fresh FP8 artifact registry malformed: {label}")
        result.append((label, path, digest))
    return result


def validate_fresh_h20_campaign_runtime_proof(
    proof: Mapping[str, Any],
    immutable_gate_quantization_check: Mapping[str, Any],
    fresh_quantization_check: Mapping[str, Any],
) -> dict:
    """CPU-pure portable validator for a fresh post-restart lifecycle envelope."""
    result = validate_seal(proof)
    required = {
        "schema_version", "kind", "status",
        "immutable_gate_quantization_check_sha256", "fresh_quantization_check_sha256",
        "runtime_equivalence", "behavior_signature", "fresh_runtime_bundle",
        "deployment", "case_set", "config", "oracle_version", "payload_sha256",
    }
    if set(result) != required:
        raise ValueError("fresh H20 campaign runtime proof field set mismatch")
    if (result.get("schema_version") != SCHEMA_VERSION
            or result.get("kind") != "h1_h20_fresh_campaign_runtime"
            or result.get("status") != "EQUIVALENT_NEW_LIFECYCLE"):
        raise ValueError("fresh H20 campaign runtime proof schema/status mismatch")
    gate_check = copy.deepcopy(dict(immutable_gate_quantization_check))
    fresh_check = copy.deepcopy(dict(fresh_quantization_check))
    if (result.get("immutable_gate_quantization_check_sha256")
            != canonical_sha256(gate_check)
            or result.get("fresh_quantization_check_sha256")
            != canonical_sha256(fresh_check)):
        raise ValueError("fresh H20 campaign runtime proof quant-check binding mismatch")
    invariant_fields = (
        "case_set", "victim", "deployment", "config", "oracle_version",
        "status", "material", "source_status",
    )
    if any(gate_check.get(field) != fresh_check.get(field) for field in invariant_fields):
        raise ValueError("fresh H20 campaign runtime immutable quant protocol mismatch")
    gate_runtime = gate_check.get("runtime_bundle")
    fresh_runtime = fresh_check.get("runtime_bundle")
    validate_h20_runtime_protocol_equivalence_document(
        result.get("runtime_equivalence"),
        immutable_gate_bundle=gate_runtime,
        fresh_runtime_bundle=fresh_runtime,
    )
    checked_fresh_runtime = validate_h20_formal_runtime_bundle(
        result.get("fresh_runtime_bundle"), require_gate_checks=False
    )
    if checked_fresh_runtime != fresh_runtime:
        raise ValueError("fresh runtime bundle copy differs from fresh quantization check")
    expected_metadata = {
        "deployment": fresh_check.get("deployment"),
        "case_set": fresh_check.get("case_set"),
        "config": fresh_check.get("config"),
        "oracle_version": fresh_check.get("oracle_version"),
    }
    if any(result.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("fresh H20 campaign runtime proof metadata mismatch")
    behavior = result.get("behavior_signature")
    behavior_key = "by_lifecycle" if "by_lifecycle" in (behavior or {}) else "by_quantization"
    if (not isinstance(behavior, Mapping)
            or behavior.get("case_set") != fresh_check.get("case_set")
            or behavior.get("sha256") != canonical_sha256(behavior.get(behavior_key))):
        raise ValueError("fresh H20 campaign behavior signature malformed")
    registry = (
        fresh_fp8_artifact_registry
        if "repeatability_path" in fresh_check else fresh_quant_artifact_registry
    )
    registry(gate_check)
    registry(fresh_check)
    return result
