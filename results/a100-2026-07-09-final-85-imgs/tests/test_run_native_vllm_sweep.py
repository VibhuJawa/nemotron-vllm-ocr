from __future__ import annotations

import base64
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_native_vllm_sweep.py"
SPEC = importlib.util.spec_from_file_location("run_native_vllm_sweep", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SWEEP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SWEEP
SPEC.loader.exec_module(SWEEP)

BENCHMARK_SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "benchmark_multi_endpoint_pooling.py"
)
BENCHMARK_SPEC = importlib.util.spec_from_file_location(
    "benchmark_multi_endpoint_pooling", BENCHMARK_SCRIPT
)
assert BENCHMARK_SPEC is not None and BENCHMARK_SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(BENCHMARK_SPEC)
sys.modules[BENCHMARK_SPEC.name] = BENCHMARK
BENCHMARK_SPEC.loader.exec_module(BENCHMARK)


class NativeVllmSweepTest(unittest.TestCase):
    def test_deep_merge_preserves_nested_defaults_without_mutation(self) -> None:
        base = {"server": {"replicas": 8, "detector": 16}, "values": [1]}
        override = {"server": {"replicas": 4}, "values": [2]}

        merged = SWEEP.deep_merge(base, override)

        self.assertEqual(
            merged,
            {"server": {"replicas": 4, "detector": 16}, "values": [2]},
        )
        self.assertEqual(base["server"]["replicas"], 8)
        merged["values"].append(3)
        self.assertEqual(override["values"], [2])

    def test_dataset_inspection_enforces_jpeg_byte_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jpeg = b"\xff\xd8\xff\xe0" + b"fake-jpeg" + b"\xff\xd9"
            uri = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
            dataset = root / "dataset.jsonl"
            rows = [
                json.dumps({"prompt": {"data": uri}}),
                json.dumps({"prompt": {"data": uri}}),
            ]
            dataset.write_text("\n".join(rows) + "\n", encoding="utf-8")

            info = SWEEP.inspect_dataset(dataset)

            self.assertEqual(info.rows, 2)
            self.assertEqual(info.jpeg_data_uri_rows, 2)
            self.assertEqual(info.jpeg_payload_bytes, len(jpeg) * 2)
            self.assertEqual(len(info.sha256), 64)

            bad = root / "bad.jsonl"
            bad.write_text(
                json.dumps(
                    {
                        "decoy": uri,
                        "prompt": {"data": "data:image/png;base64,AAAA"},
                    }
                )
                + "\n"
            )
            with self.assertRaisesRegex(SWEEP.ConfigError, "prompt.data"):
                SWEEP.inspect_dataset(bad)

    def test_client_request_keeps_canonical_model_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "dataset.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "prompt": {
                            "data": "data:image/jpeg;base64,/9j/2Q==",
                            "model": "wrong-model",
                            "truncate_prompt_tokens": 7,
                        }
                    }
                )
                + "\n"
            )

            body = json.loads(BENCHMARK.load_request_bodies(dataset, "right-model")[0])

            self.assertEqual(body["model"], "right-model")
            self.assertEqual(body["truncate_prompt_tokens"], -1)

    def test_plan_expands_replica_mps_and_native_pooling_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            normalized = SWEEP.normalize_run(
                "replica-test",
                config,
                config_dir=root,
                output_root_override=None,
                gpu_override=None,
            )
            plan = SWEEP.build_plan(
                normalized,
                invocation_dir=root / "results" / "test",
                mps_nonce="nonce",
            )

            self.assertEqual(len(plan.server_commands), 2)
            self.assertEqual(len(plan.endpoints), 2)
            self.assertEqual(
                [
                    env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"]
                    for env in plan.server_env_overrides
                ],
                ["40", "60"],
            )
            for command in plan.server_commands:
                self.assertIn("--runner", command)
                self.assertIn("pooling", command)
                self.assertIn("--renderer-num-workers", command)
                self.assertIn("--max-num-seqs", command)
                overrides = json.loads(command[command.index("--hf-overrides") + 1])
                self.assertEqual(overrides["nemotron_ocr_detector_max_batch_size"], 16)
            self.assertEqual(plan.benchmark_command.count("--endpoint"), 2)
            self.assertIn(
                "benchmark_multi_endpoint_pooling.py", plan.benchmark_command[1]
            )
            self.assertEqual(
                plan.benchmark_command[
                    plan.benchmark_command.index("--num-prompts") + 1
                ],
                "20",
            )

    def test_rejects_harness_owned_environment_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            config["env"] = {"CUDA_VISIBLE_DEVICES": "0"}
            with self.assertRaisesRegex(SWEEP.ConfigError, "harness-owned"):
                SWEEP.normalize_run(
                    "bad-env",
                    config,
                    config_dir=root,
                    output_root_override=None,
                    gpu_override=None,
                )

    def test_path_normalization_preserves_virtualenv_launcher_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            interpreter = root / "python-real"
            interpreter.touch()
            launcher = root / "venv-python"
            launcher.symlink_to(interpreter)

            resolved = SWEEP.resolve_path(str(launcher), root, "python")

            self.assertEqual(resolved, str(launcher))

    def test_process_group_cleanup_reaps_launcher_and_child(self) -> None:
        child_code = (
            "import subprocess,sys,time;"
            "subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
            "time.sleep(60)"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", child_code], start_new_session=True
        )
        try:
            time.sleep(0.1)
            SWEEP.stop_process_group(process, timeout=2)
            self.assertIsNotNone(process.poll())
            with self.assertRaises(ProcessLookupError):
                os.killpg(process.pid, signal.SIGCONT)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)

    def test_child_environment_scrubs_inherited_mps_routing(self) -> None:
        inherited = {
            "CUDA_MPS_PIPE_DIRECTORY": "/tmp/someone-elses-pipe",
            "CUDA_MPS_LOG_DIRECTORY": "/tmp/someone-elses-log",
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "7",
            "NEMOTRON_OCR_FUSED_ASPP_CONCAT": "1",
            "PYTHONPATH": "/tmp/optimized-source",
        }
        with mock.patch.dict(os.environ, inherited, clear=False):
            disabled = SWEEP.process_environment(
                {
                    "CUDA_VISIBLE_DEVICES": "5",
                    "NEMOTRON_OCR_SOURCE": "/clean/model/src",
                    "PYTHONPATH": "/clean/vllm:/clean/model/src",
                }
            )
            enabled = SWEEP.process_environment(
                {
                    "CUDA_MPS_PIPE_DIRECTORY": "/tmp/owned-pipe",
                    "CUDA_MPS_LOG_DIRECTORY": "/tmp/owned-log",
                    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "25",
                }
            )

        self.assertNotIn("CUDA_MPS_PIPE_DIRECTORY", disabled)
        self.assertNotIn("CUDA_MPS_LOG_DIRECTORY", disabled)
        self.assertNotIn("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", disabled)
        self.assertNotIn("NEMOTRON_OCR_FUSED_ASPP_CONCAT", disabled)
        self.assertEqual(disabled["NEMOTRON_OCR_SOURCE"], "/clean/model/src")
        self.assertEqual(disabled["PYTHONPATH"], "/clean/vllm:/clean/model/src")
        self.assertEqual(enabled["CUDA_MPS_PIPE_DIRECTORY"], "/tmp/owned-pipe")
        self.assertEqual(enabled["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"], "25")

    def test_gpu_selector_resolution_records_canonical_uuid(self) -> None:
        result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "5, GPU-242d3c90-db9c-a49e-e2b9-ddb0b36f1ba3, "
                "NVIDIA A100-SXM4-80GB, 00000000:C9:00.0, 81920, Default, 570.1\n"
            ),
            stderr="",
        )
        with mock.patch.object(SWEEP, "run_command", return_value=result) as command:
            gpu = SWEEP.resolve_gpu("5")

        self.assertEqual(gpu["uuid"], "GPU-242d3c90-db9c-a49e-e2b9-ddb0b36f1ba3")
        self.assertEqual(gpu["driver_version"], "570.1")
        self.assertIn("driver_version", command.call_args.args[0][3])

    def test_rejects_owned_cli_and_hf_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            server_override = self._config(root)
            server_override["server"]["extra_args"] = ["--port=9999"]
            with self.assertRaisesRegex(SWEEP.ConfigError, "harness-owned"):
                SWEEP.normalize_run(
                    "bad-server-args",
                    server_override,
                    config_dir=root,
                    output_root_override=None,
                    gpu_override=None,
                )

            benchmark_override = self._config(root)
            benchmark_override["benchmark"]["extra_args"] = [
                "--endpoint",
                "http://elsewhere",
            ]
            with self.assertRaisesRegex(SWEEP.ConfigError, "harness-owned"):
                SWEEP.normalize_run(
                    "bad-benchmark-args",
                    benchmark_override,
                    config_dir=root,
                    output_root_override=None,
                    gpu_override=None,
                )

            hf_override = self._config(root)
            hf_override["server"]["hf_overrides"] = {"nemotron_ocr_infer_length": 512}
            with self.assertRaisesRegex(SWEEP.ConfigError, "canonical OCR"):
                SWEEP.normalize_run(
                    "bad-hf-args",
                    hf_override,
                    config_dir=root,
                    output_root_override=None,
                    gpu_override=None,
                )

    def test_async_scheduling_true_is_emitted_and_nan_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            config["server"]["async_scheduling"] = True
            normalized = SWEEP.normalize_run(
                "async",
                config,
                config_dir=root,
                output_root_override=None,
                gpu_override=None,
            )
            plan = SWEEP.build_plan(
                normalized,
                invocation_dir=root / "results" / "async",
                mps_nonce="nonce",
            )
            self.assertTrue(
                all("--async-scheduling" in command for command in plan.server_commands)
            )
            self.assertTrue(
                all(
                    "--no-async-scheduling" not in command
                    for command in plan.server_commands
                )
            )

            nan_config = self._config(root)
            nan_config["ready_timeout_s"] = float("nan")
            with self.assertRaisesRegex(SWEEP.ConfigError, "positive"):
                SWEEP.normalize_run(
                    "nan-timeout",
                    nan_config,
                    config_dir=root,
                    output_root_override=None,
                    gpu_override=None,
                )

            bool_config = self._config(root)
            bool_config["server"]["mm_processor_cache_gb"] = True
            with self.assertRaisesRegex(SWEEP.ConfigError, "numeric"):
                SWEEP.normalize_run(
                    "bool-cache",
                    bool_config,
                    config_dir=root,
                    output_root_override=None,
                    gpu_override=None,
                )

    def test_git_provenance_captures_dirty_patch_and_untracked_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repo), "config", "user.name", "Test"],
                check=True,
            )
            tracked = repo / "tracked.txt"
            tracked.write_text("original\n")
            subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "commit", "-qm", "initial"], check=True
            )
            tracked.write_text("modified\n")
            (repo / "untracked.py").write_text("VALUE = 42\n")

            record = SWEEP.git_provenance(repo, root / "artifacts", "repo")

            self.assertTrue(record["dirty"])
            self.assertGreater(record["tracked_binary_patch"]["size_bytes"], 0)
            self.assertEqual(len(record["untracked_files"]), 1)
            copied = Path(record["untracked_files"][0]["captured_copy"]["path"])
            self.assertEqual(copied.read_text(), "VALUE = 42\n")

    def test_strict_source_contract_requires_pinned_clean_repositories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def initialize(name: str) -> tuple[Path, str]:
                repo = root / name
                repo.mkdir()
                subprocess.run(["git", "init", "-q", str(repo)], check=True)
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(repo),
                        "config",
                        "user.email",
                        "test@example.com",
                    ],
                    check=True,
                )
                subprocess.run(
                    ["git", "-C", str(repo), "config", "user.name", "Test"],
                    check=True,
                )
                (repo / "source.py").write_text("VALUE = 1\n")
                subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
                subprocess.run(
                    ["git", "-C", str(repo), "commit", "-qm", "initial"],
                    check=True,
                )
                commit = subprocess.run(
                    ["git", "-C", str(repo), "rev-parse", "HEAD"],
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE,
                ).stdout.strip()
                return repo, commit

            vllm, vllm_commit = initialize("vllm")
            model, model_commit = initialize("model")
            config = {
                "vllm_root": str(vllm),
                "model": str(model),
                "model_source": str(model),
                "expected_vllm_commit": vllm_commit,
                "expected_model_commit": model_commit,
                "require_clean_repositories": True,
            }

            contract = SWEEP.verify_source_contract(config)

            self.assertTrue(contract["repositories"]["vllm"]["clean"])
            self.assertTrue(contract["repositories"]["model"]["clean"])
            (model / "source.py").write_text("VALUE = 2\n")
            with self.assertRaisesRegex(SWEEP.ConfigError, "must be clean"):
                SWEEP.verify_source_contract(config)

    @staticmethod
    def _config(root: Path) -> dict[str, object]:
        for filename in ("python", "vllm", "benchmark_multi_endpoint_pooling.py"):
            (root / filename).touch()
        for dirname in ("vllm-root", "model", "model/src"):
            (root / dirname).mkdir(parents=True, exist_ok=True)
        jpeg = b"\xff\xd8\xff\xe0minimal\xff\xd9"
        data = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        (root / "dataset.jsonl").write_text(
            json.dumps({"prompt": {"data": data}}) + "\n"
        )
        return {
            "gpu": "GPU-test",
            "python": str(root / "python"),
            "vllm_cli": str(root / "vllm"),
            "vllm_root": str(root / "vllm-root"),
            "model": str(root / "model"),
            "model_source": str(root / "model/src"),
            "dataset": str(root / "dataset.jsonl"),
            "benchmark_script": str(root / "benchmark_multi_endpoint_pooling.py"),
            "output_root": str(root / "results"),
            "mps_root": str(root / "mps"),
            "env": {},
            "server": {
                "replicas": 2,
                "max_num_seqs": 64,
                "renderer_num_workers": 4,
                "detector_max_batch_size": 16,
                "mps_active_thread_percentage": [40, 60],
            },
            "benchmark": {
                "num_prompts": 20,
                "replay_count": 20,
                "concurrency_per_endpoint": 96,
            },
        }


if __name__ == "__main__":
    unittest.main()
