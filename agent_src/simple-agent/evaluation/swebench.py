import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any
import os
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from docker import from_env
from docker.errors import ImageNotFound
from tqdm import tqdm
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import asyncio

from simple_agent import CodingAgent, AgentResult, load_agent_config

logger = logging.getLogger(__name__)


def _remove_container(container):
    try:
        container.remove(force=True)
    except Exception:
        pass


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _read_instance_ids(instance_ids: list[str] | None, instance_ids_file: str | None):
    selected = list(instance_ids or [])
    if instance_ids_file:
        with open(instance_ids_file, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if line == "" or line.startswith("#"):
                    continue
                selected.append(line)
    if not selected:
        return None
    return _dedupe(selected)


def check_existing_result(query_entry) -> tuple[str, int, bool, str]:
    instance_id = query_entry["instance_id"]
    sample_id = query_entry["sample_id"]
    working_dir = query_entry["working_dir"]

    if sample_id > 0:
        result_fname = f"{instance_id}__{sample_id}.json"
    else:
        result_fname = f"{instance_id}.json"

    result_path = os.path.join(working_dir, instance_id, result_fname)
    patch_path = os.path.join(
        working_dir,
        instance_id,
        result_fname.replace(".json", ".patch"),
    )
    if os.path.exists(result_path):
        try:
            with open(result_path, "r") as f:
                result_data = json.load(f)
        except Exception:
            logger.info(
                f"Instance {instance_id}__{sample_id} has a corrupted trajectory. Re-running."
            )
            return (instance_id, sample_id, False, "corrupted")

        if result_data is None or result_data.get("final_result") is None:
            logger.info(
                f"Instance {instance_id}__{sample_id} has an incomplete trajectory. Re-running."
            )
            return (instance_id, sample_id, False, "incomplete")

        if not os.path.exists(patch_path):
            logger.info(
                f"Instance {instance_id}__{sample_id} is missing its patch. Re-running."
            )
            return (instance_id, sample_id, False, "missing_patch")
        if os.path.getsize(patch_path) == 0:
            logger.info(
                f"Instance {instance_id}__{sample_id} has an empty patch. Re-running."
            )
            return (instance_id, sample_id, False, "empty_patch")

        logger.info(
            f"Instance {instance_id}__{sample_id} already has a trajectory and patch. Skipping."
        )
        return (instance_id, sample_id, True, "complete")
    else:
        return (instance_id, sample_id, False, "missing")


class SWEBenchEvaluation:
    def __init__(
        self,
        working_dir: str,
        dataset: str = "SWE-bench_Verified",
        docker_env_config: str = "",
        swebench_harness_path: str = "",
        run_id="coding-agent",
        num_samples: int = 1,
        nproc: int = 4,
        model_config: dict[str, str] | None = None,
        instance_ids: list[str] | None = None,
    ):
        """
        Initialize the SWEBenchEvaluation class.

        Args:
            working_dir: The working directory for storing results.
            dataset: The dataset to evaluate.
            docker_env_config: The path to the docker environment config file.
            swebench_harness_path: The path to the SWEBench harness.
            run_id: The run id.
            num_samples: Number of samples per instance.
            nproc: Number of parallel processes.
            model_config: Optional dict with 'model' and 'base_url' keys.
            instance_ids: Optional selected instance ids. Filtering happens before image checks.
        """
        assert dataset in [
            "SWE-bench",
            "SWE-bench_Lite",
            "SWE-bench_Verified",
            "vmini",
            "verified_mini",
            "vtiny",
            "verified_tiny",
            "vstacktrace",
            "verified_stacktrace",
            "stacktrace",
            "swebench-debug",
            "swebench_debug",
            "debug",
        ], f"Invalid dataset name: {dataset}"
        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")
        if nproc <= 0:
            raise ValueError("nproc must be a positive integer.")

        self.dataset_arg = dataset
        verified_local_splits = {
            "vmini": "swebench_verified_mini.json",
            "verified_mini": "swebench_verified_mini.json",
            "vtiny": "swebench_verified_tiny.json",
            "verified_tiny": "swebench_verified_tiny.json",
            "vstacktrace": "swebench_verified_stacktrace.json",
            "verified_stacktrace": "swebench_verified_stacktrace.json",
            "stacktrace": "swebench_verified_stacktrace.json",
            "swebench-debug": "swebench_verified_stacktrace.json",
            "swebench_debug": "swebench_verified_stacktrace.json",
            "debug": "swebench_verified_stacktrace.json",
        }
        if dataset in verified_local_splits:
            verified_all_ds = load_dataset(
                "princeton-nlp/SWE-bench_Verified", split="test"
            )
            split_file = verified_local_splits[dataset]
            sampled = json.load(open(split_file, "r"))
            sampled_instance_ids = set([item["instance_id"] for item in sampled])
            self.dataset = verified_all_ds.filter(
                lambda x: x["instance_id"] in sampled_instance_ids
            )
            self.dataset_name = "SWE-bench_Verified"
        else:
            self.dataset = load_dataset(f"princeton-nlp/{dataset}", split="test")
            self.dataset_name = dataset

        if instance_ids:
            self.dataset = self._filter_dataset_to_instances(instance_ids)

        self.docker_client = from_env()
        self.image_status: dict[Any, Any] = {}
        self.working_dir = Path(working_dir)
        self.swebench_harness_path = swebench_harness_path
        self.run_id = run_id
        self.model_config = model_config or {}
        self.num_samples = num_samples
        self.nproc = nproc

        if docker_env_config != "":
            with open(docker_env_config, "r") as f:
                self.docker_env_config = json.load(f)
        else:
            self.docker_env_config = {}

        if not self.working_dir.exists():
            self.working_dir.mkdir(parents=True, exist_ok=True)

        self.instances = [dict(instance) for instance in self.dataset]
        self.instances_by_id = {
            instance["instance_id"]: instance for instance in self.instances
        }
        self.status_log_path = self.working_dir / "run_status.jsonl"
        self.summary_path = self.working_dir / "run_summary.json"
        self.progress_counts: dict[str, int] = {}

        self.pull_images()
        self.pbar = None

    def _filter_dataset_to_instances(self, instance_ids: list[str]):
        requested = list(dict.fromkeys(instance_ids))
        available = {instance["instance_id"] for instance in self.dataset}
        missing = [instance_id for instance_id in requested if instance_id not in available]
        if missing:
            raise ValueError(f"Unknown instance ids for selected dataset: {missing}")
        requested_set = set(requested)
        return self.dataset.filter(lambda x: x["instance_id"] in requested_set)

    def _image_name(self, instance_id: str) -> str:
        key = f"swebench/sweb.eval.x86_64.{instance_id.lower()}:latest"
        key = key.replace("__", "_1776_")
        return key

    def _check_images(self):
        for instance in tqdm(self.instances, desc="Checking image status"):
            instance_id = instance["instance_id"]
            image_name = self._image_name(instance_id)
            try:
                _ = self.docker_client.images.get(image_name)
                self.image_status[instance_id] = True
            except ImageNotFound:
                self.image_status[instance_id] = False

    def pull_images(self):
        self._check_images()
        print(f"Total number of images: {len(self.image_status)}")
        instance_ids = [
            instance_id
            for instance_id in self.image_status
            if not self.image_status[instance_id]
        ]
        print(f"Number of images to download: {len(instance_ids)}")
        if len(instance_ids) == 0:
            return

        async def pull_one(instance_id) -> bool:
            image_name = self._image_name(instance_id)
            try:
                await asyncio.to_thread(self.docker_client.images.pull, image_name)
                print(f"Pulled image: {image_name}")
                return True
            except Exception as e:
                print(f"Failed to pull image {image_name}: {e}")
                return False

        async def async_pull_images(instance_ids):
            semaphore = asyncio.Semaphore(min(self.nproc, 8))
            completed = 0
            failed = 0

            async def guarded_pull(instance_id):
                nonlocal completed, failed
                async with semaphore:
                    ok = await pull_one(instance_id)
                    completed += 1
                    if not ok:
                        failed += 1
                    self.pbar.set_postfix_str(f"completed={completed} failed={failed}")
                    self.pbar.update(1)
                    self.pbar.refresh()

            self.pbar = tqdm(
                total=len(instance_ids),
                desc="Pulling Docker images",
                unit="image",
            )
            tasks = [asyncio.create_task(guarded_pull(instance_id)) for instance_id in instance_ids]
            await asyncio.gather(*tasks)

        asyncio.run(async_pull_images(instance_ids))

    def prepare_container(self, instance):
        """
        Prepare a Docker container for a given instance.

        Args:
            instance: A dictionary containing instance information.

        Returns:
            The Docker container object.
        """
        image_name = self._image_name(instance["instance_id"])

        instance_dir = self.working_dir / instance["instance_id"]
        instance_dir.mkdir(parents=True, exist_ok=True)

        container = self.docker_client.containers.run(
            image_name,
            command="/bin/bash",
            detach=True,
            tty=True,
            stdin_open=True,
            working_dir="/testbed",
            environment=self.docker_env_config.get("experiment_env", None),
        )

        return container

    def _instance_prefix(self, instance_id: str, sample_id: int) -> str:
        """Return the file-name prefix for a given instance and sample."""
        if sample_id > 0:
            return f"{instance_id}__{sample_id}"
        return instance_id

    def _save_result(self, instance_id: str, sample_id: int, result: AgentResult):
        """Save the patch to disk. The trajectory JSON is already written by the agent."""
        instance_dir = self.working_dir / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)

        prefix = self._instance_prefix(instance_id, sample_id)
        patch_path = instance_dir / f"{prefix}.patch"

        with open(patch_path, "w") as f:
            f.write(result.diff)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _patch_path(self, instance_id: str, sample_id: int) -> Path:
        prefix = self._instance_prefix(instance_id, sample_id)
        return self.working_dir / instance_id / f"{prefix}.patch"

    def _trajectory_path(self, instance_id: str, sample_id: int) -> Path:
        prefix = self._instance_prefix(instance_id, sample_id)
        return self.working_dir / instance_id / f"{prefix}.json"

    def _append_status(self, record: dict[str, Any]):
        with open(self.status_log_path, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def _write_summary(self):
        summary = {
            "dataset_arg": self.dataset_arg,
            "dataset_name": self.dataset_name,
            "run_id": self.run_id,
            "working_dir": self.working_dir.as_posix(),
            "selected_instance_count": len(self.instances),
            "num_samples": self.num_samples,
            "nproc": self.nproc,
            "updated_at": self._now(),
            "counts": dict(self.progress_counts),
        }
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")

    def _status_record(
        self,
        instance_id: str,
        sample_id: int,
        status: str,
        started_at: str | None = None,
        ended_at: str | None = None,
        elapsed_seconds: float | None = None,
        error_summary: str | None = None,
        resume_reason: str | None = None,
        return_code: int | None = None,
    ) -> dict[str, Any]:
        return {
            "instance_id": instance_id,
            "sample_id": sample_id,
            "status": status,
            "started_at": started_at,
            "ended_at": ended_at,
            "elapsed_seconds": elapsed_seconds,
            "patch_path": self._patch_path(instance_id, sample_id).as_posix(),
            "trajectory_path": self._trajectory_path(instance_id, sample_id).as_posix(),
            "return_code": return_code,
            "error_summary": error_summary,
            "resume_reason": resume_reason,
        }

    def run_one_instance(self, instance_id):
        """
        Run a single instance.

        Args:
            instance_id: The ID of the instance to run.
        """
        instance = None
        for inst in self.dataset:
            if inst["instance_id"] == instance_id:
                instance = inst
        if instance is None:
            print(f"Instance {instance_id} not found.")
            return

        for sample_id in range(self.num_samples):
            instance_dir = self.working_dir / instance_id
            prefix = self._instance_prefix(instance_id, sample_id)
            trajectory_path = str(instance_dir / f"{prefix}.json")
            container = None
            try:
                container = self.prepare_container(instance)
                agent = CodingAgent(container, **self.model_config)
                result = agent.run(instance["problem_statement"], trajectory_path=trajectory_path)
                self._save_result(instance_id, sample_id, result)
                print(
                    f"Instance {instance_id} sample {sample_id} completed with status: {result.status}"
                )
            except Exception:
                print(f"Instance {instance_id} sample {sample_id} failed.")
                print(traceback.format_exc())
            finally:
                if container is not None:
                    threading.Thread(
                        target=lambda c=container: _remove_container(c), daemon=True
                    ).start()

    def run_sample_job(self, instance_id, sample_id, resume_reason: str | None = None):
        instance = self.instances_by_id.get(instance_id)
        if instance is None:
            return self._status_record(
                instance_id,
                sample_id,
                "failed",
                started_at=self._now(),
                ended_at=self._now(),
                elapsed_seconds=0,
                error_summary=f"Instance {instance_id} not found.",
                resume_reason=resume_reason,
            )

        started_at = self._now()
        started = time.monotonic()
        trajectory_path = self._trajectory_path(instance_id, sample_id)
        container = None
        try:
            container = self.prepare_container(instance)
            agent = CodingAgent(container, **self.model_config)
            result = agent.run(instance["problem_statement"], trajectory_path=str(trajectory_path))
            self._save_result(instance_id, sample_id, result)
            ended_at = self._now()
            status = "completed" if result.status != "fail" else "failed"
            print(f"Instance {instance_id}__{sample_id} completed with status: {result.status}")
            return self._status_record(
                instance_id,
                sample_id,
                status,
                started_at=started_at,
                ended_at=ended_at,
                elapsed_seconds=round(time.monotonic() - started, 3),
                error_summary=None if status == "completed" else result.status,
                resume_reason=resume_reason,
                return_code=0 if status == "completed" else 1,
            )
        except Exception as exc:
            print(f"Instance {instance_id}__{sample_id} failed.")
            print(traceback.format_exc())
            return self._status_record(
                instance_id,
                sample_id,
                "failed",
                started_at=started_at,
                ended_at=self._now(),
                elapsed_seconds=round(time.monotonic() - started, 3),
                error_summary=str(exc)[:1000],
                resume_reason=resume_reason,
                return_code=1,
            )
        finally:
            if container is not None:
                threading.Thread(
                    target=lambda c=container: _remove_container(c), daemon=True
                ).start()

    async def async_run_one_instance(self, instance_id, sample_id, resume_reason=None):
        """
        Run a single instance asynchronously.

        Args:
            instance_id: The ID of the instance to run.
            sample_id: The sample ID.
        """
        return await asyncio.to_thread(
            self.run_sample_job, instance_id, sample_id, resume_reason
        )

    def run_all(self, instance_ids: list[str] | None = None):
        """
        Run all instances in the dataset.
        """
        instances = [
            self.instances_by_id[instance_id]
            for instance_id in instance_ids
        ] if instance_ids else self.instances
        to_query = []
        for instance in instances:
            for sample_id in range(self.num_samples):
                to_query.append(
                    {
                        "instance_id": instance["instance_id"],
                        "sample_id": sample_id,
                        "working_dir": str(self.working_dir),
                    }
                )

        # Check existing results using multiprocessing
        should_skip_rets = []
        with Pool(processes=min(48, len(to_query))) as pool:
            results = pool.imap_unordered(check_existing_result, to_query)
            for ret in tqdm(
                results, total=len(to_query), desc="Checking existing results"
            ):
                should_skip_rets.append(ret)

        should_skip = set()
        resume_reasons = {}
        for instance_id, sample_id, skip, reason in should_skip_rets:
            resume_reasons[(instance_id, sample_id)] = reason
            if skip:
                should_skip.add((instance_id, sample_id))
        print(f"Skipping {len(should_skip)} already completed instances.")

        self.progress_counts = {
            "planned": len(to_query),
            "skipped": len(should_skip),
            "runnable": len(to_query) - len(should_skip),
            "running": 0,
            "completed": 0,
            "failed": 0,
            "rerun": sum(
                1 for key, reason in resume_reasons.items()
                if key not in should_skip
                and reason in {"corrupted", "incomplete", "missing_patch", "empty_patch"}
            ),
        }
        self._write_summary()
        print("SWE-bench Verified simple-agent run summary")
        print(f"  dataset: {self.dataset_name}")
        print(f"  selected instances: {len(instances)}")
        print(f"  samples: {self.num_samples}")
        print(f"  planned jobs: {self.progress_counts['planned']}")
        print(f"  skipped jobs: {self.progress_counts['skipped']}")
        print(f"  runnable jobs: {self.progress_counts['runnable']}")
        print(f"  nproc: {self.nproc}")
        print(f"  working dir: {self.working_dir}")

        for instance_id, sample_id in sorted(should_skip):
            record = self._status_record(
                instance_id,
                sample_id,
                "skipped",
                resume_reason=resume_reasons.get((instance_id, sample_id)),
            )
            self._append_status(record)

        async def _run_all_instances():
            semaphore = asyncio.Semaphore(self.nproc)
            self.pbar = tqdm(
                total=self.progress_counts["runnable"],
                desc="Running all instances",
                unit="instance",
            )

            async def run_guarded(instance_id, sample_id):
                async with semaphore:
                    self.progress_counts["running"] += 1
                    self._write_summary()
                    record = await self.async_run_one_instance(
                        instance_id,
                        sample_id,
                        resume_reasons.get((instance_id, sample_id)),
                    )
                    self.progress_counts["running"] -= 1
                    if record["status"] == "completed":
                        self.progress_counts["completed"] += 1
                    else:
                        self.progress_counts["failed"] += 1
                    self._append_status(record)
                    self._write_summary()
                    self.pbar.update(1)
                    self.pbar.set_postfix(self.progress_counts)
                    self.pbar.refresh()

            tasks = []
            for instance in instances:
                for sample_id in range(self.num_samples):
                    if (instance["instance_id"], sample_id) in should_skip:
                        continue
                    tasks.append(
                        asyncio.create_task(
                            run_guarded(instance["instance_id"], sample_id)
                        )
                    )
            await asyncio.gather(*tasks)

        asyncio.run(_run_all_instances())

    def run_eval(self, sample_id=0):
        """
        Run evaluation using the SWE-bench harness.
        """
        swebench_harness_path = Path(self.swebench_harness_path)
        swebench_python_path = "swebench_venv/bin/python"
        if sample_id == 0:
            suffix = "predictions.json"
            run_id = self.run_id
        else:
            suffix = f"predictions_{sample_id}.json"
            run_id = f"{self.run_id}_{sample_id}"

        cmd = [
            swebench_python_path,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            f"princeton-nlp/{self.dataset_name}",
            "--predictions_path",
            (self.working_dir / suffix).absolute().as_posix(),
            "--run_id",
            run_id,
            "--cache_level",
            "instance",
            "--instance_image_tag",
            "latest",
        ]
        print(f"Running evaluation with command: {' '.join(cmd)}")
        process = subprocess.run(
            cmd, capture_output=True, cwd=swebench_harness_path.as_posix()
        )
        print(process.stdout.decode())
        print(process.stderr.decode())

        result_filename = f"{self.run_id}.{run_id}.json"
        print(f"Evaluation completed and file saved to {result_filename}")

    def get_all_preds(
        self, instance_ids: list[str] | None = None, sample_id: int = 0
    ):
        """
        Collect all predictions into a single file for evaluation.

        Args:
            instance_ids: A list of instance IDs. If None, all instances in the dataset will be used.
            sample_id: The sample ID to collect predictions for.
        """
        preds = []
        if not instance_ids:
            instance_ids = [instance["instance_id"] for instance in self.dataset]
        for instance_id in instance_ids:
            if sample_id > 0:
                patch_path = (
                    self.working_dir / instance_id / f"{instance_id}__{sample_id}.patch"
                )
            else:
                patch_path = self.working_dir / instance_id / f"{instance_id}.patch"
            if not patch_path.exists():
                continue
            with open(patch_path, "r") as f:
                patch = f.read()
            preds.append(
                {
                    "instance_id": instance_id,
                    "model_name_or_path": self.run_id,
                    "model_patch": patch,
                }
            )
        if sample_id > 0:
            preds_path = self.working_dir / f"predictions_{sample_id}.json"
        else:
            preds_path = self.working_dir / "predictions.json"
        with open(preds_path, "w") as f:
            json.dump(preds, f)
        print(f"Collected {len(preds)} predictions to {preds_path}")


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--dataset", type=str, default="SWE-bench_Verified"
    )
    argument_parser.add_argument(
        "--working-dir", type=str, default="./workspace"
    )
    argument_parser.add_argument(
        "--instance_ids",
        nargs="+",
        type=str,
        help="Instance IDs to run (space separated)",
    )
    argument_parser.add_argument(
        "--instance-ids-file",
        type=str,
        default=None,
        help="File containing instance IDs to run, one per line. Blank lines and # comments are ignored.",
    )
    argument_parser.add_argument(
        "--swebench-harness-path",
        type=str,
        default="",
        required=False,
        help="Only used for evaluation.",
    )
    argument_parser.add_argument(
        "--docker-env-config", type=str, default="", required=False
    )
    argument_parser.add_argument(
        "--run-id",
        type=str,
        required=False,
        default="coding-agent",
        help="Run ID for SWE-bench evaluation.",
    )
    argument_parser.add_argument(
        "--num_samples",
        type=int,
        required=False,
        default=1,
        help="Number of samples for each instance.",
    )
    argument_parser.add_argument(
        "--nproc",
        type=int,
        required=False,
        default=4,
        help="Number of parallel processes.",
    )
    # expr: only generate patches
    # eval: only evaluate patches
    # e2e: both expr and eval
    argument_parser.add_argument(
        "--mode",
        type=str,
        choices=["e2e", "expr", "eval"],
        default="e2e",
        help="e2e: both expr and eval, expr: only generate patches, eval: only evaluate patches",
    )
    argument_parser.add_argument(
        "--agent-config",
        type=str,
        required=False,
        default=None,
        help="Path to agent_configs.yaml file.",
    )
    argument_parser.add_argument(
        "--provider",
        type=str,
        required=False,
        default=None,
        help="Agent config name to use from the config file.",
    )

    args = argument_parser.parse_args()

    model_config = load_agent_config(args.agent_config, args.provider) if args.agent_config else {}
    selected_instance_ids = _read_instance_ids(args.instance_ids, args.instance_ids_file)

    evaluation = SWEBenchEvaluation(
        args.working_dir,
        args.dataset,
        args.docker_env_config,
        args.swebench_harness_path,
        args.run_id,
        args.num_samples,
        args.nproc,
        model_config=model_config,
        instance_ids=selected_instance_ids,
    )

    if args.mode in ("e2e", "expr"):
        if selected_instance_ids:
            print(f"Running instances: {selected_instance_ids}")
            evaluation.run_all(selected_instance_ids)
        else:
            print("Running all instances")
            evaluation.run_all()

    if args.mode in ("e2e", "eval"):
        for sample_id in range(args.num_samples):
            evaluation.get_all_preds(selected_instance_ids, sample_id=sample_id)
            evaluation.run_eval(sample_id=sample_id)


if __name__ == "__main__":
    main()
