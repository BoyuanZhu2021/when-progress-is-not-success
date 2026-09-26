"""Goldens for the victim launcher preflight.

Each test here corresponds to a way a real campaign was lost or nearly lost. They run on CPU with
no GPU and no server.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import serve_victim  # noqa: E402
from serve_victim import (  # noqa: E402
    PreflightError,
    build_serve_argv,
    launch_command,
    model_family,
    parser_for_model,
    resolve_local_snapshot,
    validate_quant,
    victim_identity,
    canonical_manifest,
)


class TestParserSelection(unittest.TestCase):
    def test_qwen35_beats_qwen3_prefix(self):
        """The bug that zeroed a campaign: 'Qwen3.5-27B' also startswith 'qwen3'. Longest match
        must win, or a Qwen3.5 victim silently gets the hermes parser."""
        self.assertEqual(model_family("Qwen/Qwen3.5-27B"), "qwen3.5")
        self.assertEqual(parser_for_model("Qwen/Qwen3.5-27B"), "qwen3_xml")
        self.assertEqual(parser_for_model("Qwen/Qwen3.5-9B"), "qwen3_xml")

    def test_qwen38_does_not_fall_through_to_hermes(self):
        """Qwen3.8 shares Qwen3.5's architecture and XML tool-call format, but "qwen3.8-27b"
        also startswith "qwen3" -- without an explicit entry it silently gets hermes and phi
        collapses to 0 for the whole campaign."""
        self.assertEqual(model_family("Qwen/Qwen3.8-27B"), "qwen3.8")
        self.assertEqual(parser_for_model("Qwen/Qwen3.8-27B"), "qwen3_xml")
        self.assertEqual(parser_for_model("Qwen/Qwen3.8-27B-FP8"), "qwen3_xml")

    def test_other_families_keep_hermes(self):
        self.assertEqual(parser_for_model("Qwen/Qwen3-32B"), "hermes")
        self.assertEqual(parser_for_model("Qwen/Qwen2.5-32B-Instruct"), "hermes")

    def test_unknown_family_fails_closed(self):
        """Must raise, NOT default to hermes -- defaulting is what produced tool_calls=null."""
        with self.assertRaises(PreflightError) as cm:
            parser_for_model("meta-llama/Llama-3-70B")
        self.assertIn("Refusing to guess", str(cm.exception))

    def test_bare_name_without_org(self):
        self.assertEqual(parser_for_model("Qwen3.5-27B"), "qwen3_xml")


class TestQuantValidation(unittest.TestCase):
    def test_fp8_refused_on_ampere(self):
        """A800/A100 are sm80 and have no FP8; the last campaign had to serve bf16 (deviation D1)."""
        with self.assertRaises(PreflightError) as cm:
            validate_quant("fp8", (8, 0))
        self.assertIn("NO FP8", str(cm.exception))

    def test_fp8_allowed_on_hopper_and_ada(self):
        validate_quant("fp8", (9, 0))   # H20 / H100
        validate_quant("fp8", (8, 9))   # Ada, the documented floor

    def test_bf16_always_allowed(self):
        validate_quant("bf16", (8, 0))
        validate_quant("bf16", (9, 0))
        validate_quant("bf16", None)

    def test_unknown_capability_does_not_block(self):
        """nvidia-smi can be absent (e.g. --plan from a laptop); don't hard-fail on that."""
        validate_quant("fp8", None)

    def test_dtype_auto_with_prequantized(self):
        argv = build_serve_argv("Qwen/Qwen3.8-27B-FP8", quant="auto",
                                tool_call_parser="qwen3_xml")
        self.assertEqual(argv[argv.index("--dtype") + 1], "auto")

    def test_dtype_bf16_otherwise(self):
        for q in ("bf16", "fp8"):
            argv = build_serve_argv("Qwen/Qwen3.8-27B", quant=q, tool_call_parser="qwen3_xml")
            self.assertEqual(argv[argv.index("--dtype") + 1], "bfloat16")

    def test_auto_omits_quantization_flag(self):
        """A pre-quantized checkpoint already carries quantization_config; forcing
        --quantization fp8 on top of it is wrong. auto lets vLLM read the checkpoint."""
        argv = build_serve_argv("Qwen/Qwen3.8-27B-FP8", quant="auto",
                                tool_call_parser="qwen3_xml")
        self.assertNotIn("--quantization", argv)

    def test_prequantized_model_still_needs_fp8_hardware(self):
        """The capability guard must not be bypassable by just not naming the precision."""
        with self.assertRaises(PreflightError):
            validate_quant("auto", (8, 0), "Qwen/Qwen3.8-27B-FP8")
        validate_quant("auto", (9, 0), "Qwen/Qwen3.8-27B-FP8")
        validate_quant("auto", (8, 0), "Qwen/Qwen3.8-27B")   # bf16 repo is fine on Ampere

    def test_bad_quant_name(self):
        with self.assertRaises(PreflightError):
            validate_quant("int4", (9, 0))


class TestServeArgv(unittest.TestCase):
    def test_tool_flags_always_present(self):
        """Their absence 400s every victim call. They are not optional, for any model or quant."""
        for model in ("Qwen/Qwen3.5-27B", "Qwen/Qwen3-32B", "Qwen/Qwen3.5-9B"):
            for quant in ("bf16", "fp8"):
                argv = build_serve_argv(model, quant=quant)
                self.assertIn("--enable-auto-tool-choice", argv, f"{model}/{quant}")
                self.assertIn("--tool-call-parser", argv, f"{model}/{quant}")
                self.assertTrue(argv[argv.index("--tool-call-parser") + 1])

    def test_parser_value_matches_family(self):
        argv = build_serve_argv("Qwen/Qwen3.5-27B")
        self.assertEqual(argv[argv.index("--tool-call-parser") + 1], "qwen3_xml")

    def test_fp8_only_when_asked(self):
        self.assertNotIn("--quantization", build_serve_argv("Qwen/Qwen3.5-27B", quant="bf16"))
        argv = build_serve_argv("Qwen/Qwen3.5-27B", quant="fp8")
        self.assertEqual(argv[argv.index("--quantization") + 1], "fp8")

    def test_revision_appended_when_given(self):
        self.assertNotIn("--revision", build_serve_argv("Qwen/Qwen3.5-27B"))
        argv = build_serve_argv("Qwen/Qwen3.5-27B", revision="abc123")
        self.assertEqual(argv[argv.index("--revision") + 1], "abc123")

    def test_tensor_parallel_reflected(self):
        argv = build_serve_argv("Qwen/Qwen3.5-27B", tp=2)
        self.assertEqual(argv[argv.index("--tensor-parallel-size") + 1], "2")

    def test_override_parser_wins(self):
        argv = build_serve_argv("Qwen/Qwen3.5-27B", tool_call_parser="hermes")
        self.assertEqual(argv[argv.index("--tool-call-parser") + 1], "hermes")

    def test_unknown_model_still_fails_closed_through_argv(self):
        with self.assertRaises(PreflightError):
            build_serve_argv("mistralai/Mistral-7B")


class TestLaunchCommand(unittest.TestCase):
    """Regression: the first launch attempt died with
    `nohup: failed to run command 'CUDA_VISIBLE_DEVICES=1'` because a bare VAR=x prefix is shell
    syntax, not an executable, and nohup execs its first argument."""

    def test_first_token_is_an_executable_not_an_assignment(self):
        cmd = launch_command(["python", "-m", "vllm"], {"CUDA_VISIBLE_DEVICES": "1"})
        first = cmd.split()[0]
        self.assertEqual(first, "env")
        self.assertNotIn("=", first)

    def test_env_and_argv_both_survive(self):
        cmd = launch_command(["python", "-m", "vllm", "--port", "8000"],
                             {"CUDA_VISIBLE_DEVICES": "0", "HF_HUB_DISABLE_XET": "1"})
        self.assertIn("CUDA_VISIBLE_DEVICES=0", cmd)
        self.assertIn("HF_HUB_DISABLE_XET=1", cmd)
        self.assertIn("--port 8000", cmd)

    def test_values_with_spaces_are_quoted(self):
        cmd = launch_command(["python"], {"X": "a b"})
        self.assertIn("X='a b'", cmd)

    def test_local_path_needs_explicit_parser(self):
        """Regression: once --model is replaced by a snapshot PATH, family inference is impossible;
        the caller must pass the parser resolved from the repo id."""
        path = "/root/hf/hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec"
        with self.assertRaises(PreflightError):
            build_serve_argv(path)
        argv = build_serve_argv(path, tool_call_parser="qwen3_xml",
                                served_name="Qwen/Qwen3.5-27B")
        self.assertEqual(argv[argv.index("--model") + 1], path)
        self.assertEqual(argv[argv.index("--served-model-name") + 1], "Qwen/Qwen3.5-27B")
        self.assertEqual(argv[argv.index("--tool-call-parser") + 1], "qwen3_xml")

    def test_interpreter_bin_goes_on_path(self):
        """Regression: engine init died with FileNotFoundError: 'ninja'. vLLM shells out to ninja
        for JIT kernels, and calling a conda env's python by absolute path does not put that env's
        bin/ on PATH."""
        import os
        from pathlib import Path as P
        env_bin = str(P("/opt/conda/envs/vllm/bin"))
        cmd = launch_command(["/opt/conda/envs/vllm/bin/python", "-m", "vllm"],
                             {"PATH": env_bin + os.pathsep + "/usr/bin"})
        self.assertIn("/opt/conda/envs/vllm/bin", cmd)
        self.assertTrue(cmd.split("PATH=")[1].lstrip("'\"").startswith(env_bin),
                        "env bin must come FIRST on PATH")


class TestResolveLocalSnapshot(unittest.TestCase):
    """transformers 5.13 cannot resolve a repo id from the HF cache offline on this box (verified
    in both conda envs, including the one that downloaded the weights), so vLLM must be handed the
    snapshot directory."""

    def _cache(self, root, repo, revs):
        d = Path(root) / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots"
        for r in revs:
            (d / r).mkdir(parents=True)
            (d / r / "config.json").write_text("{}", encoding="utf-8")
        return d

    def test_none_when_not_cached(self):
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            self.assertIsNone(resolve_local_snapshot("Qwen/Qwen3.5-27B", hf_home=t))

    def test_returns_snapshot_dir(self):
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            self._cache(t, "Qwen/Qwen3.5-27B", ["fc05daec"])
            got = resolve_local_snapshot("Qwen/Qwen3.5-27B", hf_home=t)
            self.assertTrue(got.endswith("fc05daec"))

    def test_explicit_revision_selected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            self._cache(t, "Qwen/Qwen3.5-27B", ["aaa", "bbb"])
            self.assertTrue(resolve_local_snapshot("Qwen/Qwen3.5-27B", "bbb", hf_home=t)
                            .endswith("bbb"))

    def test_ambiguous_revisions_refused(self):
        """Two cached revisions and no --revision must stop, not silently pick one: 'which weights
        actually ran' is not a coin flip."""
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            self._cache(t, "Qwen/Qwen3.5-27B", ["aaa", "bbb"])
            with self.assertRaises(PreflightError) as cm:
                resolve_local_snapshot("Qwen/Qwen3.5-27B", hf_home=t)
            self.assertIn("Refusing to guess", str(cm.exception))

    def test_incomplete_snapshot_ignored(self):
        """A snapshot dir without config.json is a half-finished download, not a servable model."""
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "hub" / "models--Qwen--Qwen3.5-27B" / "snapshots" / "aaa"
            d.mkdir(parents=True)
            self.assertIsNone(resolve_local_snapshot("Qwen/Qwen3.5-27B", hf_home=t))


class TestVictimIdentity(unittest.TestCase):
    """run_meta must record HOW the victim was served, and must never die trying."""

    def test_never_raises_on_dead_endpoint_and_missing_manifest(self):
        """A metadata gap must not kill a training run that is otherwise fine."""
        got = victim_identity("http://127.0.0.1:9/v1", manifest_path="/nonexistent/x.json",
                              timeout=1)
        self.assertEqual(got["base_url"], "http://127.0.0.1:9/v1")

    def test_absence_is_explicit_not_omitted(self):
        """null distinguishes 'not served by our launcher' from 'nobody wrote it down' -- the
        exact ambiguity that made the last campaign's serve config unrecoverable."""
        got = victim_identity("http://127.0.0.1:9/v1", manifest_path="/nonexistent/x.json",
                              timeout=1)
        self.assertIn("serve_manifest", got)
        self.assertIn("served_models", got)
        self.assertIsNone(got["serve_manifest"])

    def test_reads_manifest_when_present(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "victim_serve.json"
            p.write_text(json.dumps({
                "model": "Qwen/Qwen3.5-27B", "revision": None, "quant": "fp8", "tp": 1,
                "tool_call_parser": "qwen3_xml", "compute_capability": "9.0",
                "extra_noise": "ignored"}), encoding="utf-8")
            got = victim_identity("http://127.0.0.1:9/v1", manifest_path=str(p), timeout=1)
        self.assertEqual(got["serve_manifest"]["model"], "Qwen/Qwen3.5-27B")
        self.assertEqual(got["serve_manifest"]["quant"], "fp8")
        self.assertEqual(got["serve_manifest"]["tool_call_parser"], "qwen3_xml")
        self.assertEqual(got["serve_manifest"]["compute_capability"], "9.0")
        self.assertNotIn("extra_noise", got["serve_manifest"])


class ManifestProvenanceTest(unittest.TestCase):
    """G0 (EXP-2026W36-002) recorded `Qwen3.8-27B-FP8` in run_meta for runs that actually talked to a
    9B: `victim_identity` read one FIXED path while four victims each wrote its own --manifest, so
    every lane inherited whatever last touched the default. These goldens pin the per-port fix."""

    def test_canonical_manifest_is_keyed_by_port(self):
        self.assertNotEqual(canonical_manifest(8000), canonical_manifest(8001))
        self.assertIn("8002", canonical_manifest(8002))
        self.assertEqual(canonical_manifest(8003), canonical_manifest("8003"))

    def test_identity_resolves_the_port_not_the_default(self):
        """The exact G0 failure: a stale default plus a correct per-port file must yield the port's."""
        with tempfile.TemporaryDirectory() as td:
            stale = Path(td) / "default.json"
            stale.write_text(json.dumps({"model": "Qwen/Qwen3.8-27B-FP8", "quant": "auto"}), encoding="utf-8")
            fresh = Path(td) / "port.json"
            fresh.write_text(json.dumps({"model": "Qwen/Qwen3.5-9B", "quant": "bf16"}), encoding="utf-8")
            with mock.patch.object(serve_victim, "DEFAULT_MANIFEST", str(stale)),                  mock.patch.object(serve_victim, "canonical_manifest", lambda port: str(fresh)):
                got = serve_victim.victim_identity("http://127.0.0.1:8001/v1", timeout=0)
        self.assertEqual(got["serve_manifest"]["model"], "Qwen/Qwen3.5-9B")
        self.assertEqual(got["serve_manifest"]["quant"], "bf16")
        self.assertEqual(got["serve_manifest_path"], str(fresh))

    def test_identity_falls_back_to_default_for_pre_fix_runs(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "default.json"
            d.write_text(json.dumps({"model": "Qwen/Qwen3.5-9B"}), encoding="utf-8")
            with mock.patch.object(serve_victim, "DEFAULT_MANIFEST", str(d)),                  mock.patch.object(serve_victim, "canonical_manifest", lambda port: str(Path(td) / "absent.json")):
                got = serve_victim.victim_identity("http://127.0.0.1:8001/v1", timeout=0)
        self.assertEqual(got["serve_manifest"]["model"], "Qwen/Qwen3.5-9B")
        self.assertEqual(got["serve_manifest_path"], str(d))

    def test_identity_never_raises_and_marks_absence_explicitly(self):
        with tempfile.TemporaryDirectory() as td:
            miss = str(Path(td) / "nope.json")
            with mock.patch.object(serve_victim, "DEFAULT_MANIFEST", miss),                  mock.patch.object(serve_victim, "canonical_manifest", lambda port: miss):
                got = serve_victim.victim_identity("not-a-url", timeout=0)
        self.assertIsNone(got["serve_manifest"])
        self.assertIsNone(got["serve_manifest_path"])
        self.assertEqual(got["base_url"], "not-a-url")

    def test_explicit_missing_path_does_not_fall_back(self):
        """An explicitly named manifest that is absent must yield None, NOT some other file.
        The first version of the fallback chain regressed exactly this and only failed on the box,
        where DEFAULT_MANIFEST held the previous campaign's 27B entry."""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "default.json"
            d.write_text(json.dumps({"model": "Qwen/Qwen3.8-27B-FP8"}), encoding="utf-8")
            with mock.patch.object(serve_victim, "DEFAULT_MANIFEST", str(d)),                  mock.patch.object(serve_victim, "canonical_manifest", lambda port: str(d)):
                got = serve_victim.victim_identity("http://127.0.0.1:8001/v1",
                                                   manifest_path=str(Path(td) / "absent.json"), timeout=0)
        self.assertIsNone(got["serve_manifest"])
        self.assertIsNone(got["serve_manifest_path"])

    def test_explicit_override_wins_over_both(self):
        with tempfile.TemporaryDirectory() as td:
            for nm, model in (("d.json", "D"), ("p.json", "P"), ("x.json", "X")):
                (Path(td)/nm).write_text(json.dumps({"model": model}), encoding="utf-8")
            with mock.patch.object(serve_victim, "DEFAULT_MANIFEST", str(Path(td)/"d.json")),                  mock.patch.object(serve_victim, "canonical_manifest", lambda port: str(Path(td)/"p.json")):
                got = serve_victim.victim_identity("http://127.0.0.1:8001/v1",
                                                   manifest_path=str(Path(td)/"x.json"), timeout=0)
        self.assertEqual(got["serve_manifest"]["model"], "X")


if __name__ == "__main__":
    unittest.main(verbosity=2)
