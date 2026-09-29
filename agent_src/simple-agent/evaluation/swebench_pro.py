# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import argparse
import asyncio
import csv
import json
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset
from docker import from_env
from docker.errors import ImageNotFound
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_agent import CodingAgent, AgentResult, load_agent_config, load_config


EVAL_DIR = Path(__file__).resolve().parent


def docker_exec(container, command: str):
    exec_result = container.exec_run(cmd=command)
    return_code = exec_result[0]
    output = exec_result[1].decode("utf-8", errors="replace")
    return return_code, output


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tail(text: str | None, max_chars: int = 2000) -> str | None:
    if text is None:
        return None
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _remove_container(container):
    try:
        container.stop(timeout=1)
    except Exception:
        pass
    try:
        container.remove(force=True)
    except Exception:
        pass


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _resolve_eval_path(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        return path
    eval_path = EVAL_DIR / path
    if eval_path.exists():
        return eval_path
    trae_eval_path = EVAL_DIR.parent.parent / "trae-agent" / "evaluation" / path
    if trae_eval_path.exists():
        return trae_eval_path
    return path


def _dockerhub_image_uri(instance: dict[str, Any], dockerhub_username: str) -> str:
    if instance.get("dockerhub_tag"):
        return f"{dockerhub_username}/sweap-images:{instance['dockerhub_tag']}"

    uid = instance["instance_id"]
    repo_base, repo_name_only = instance["repo"].lower().split("/")
    hsh = uid.replace("instance_", "")
    if uid == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan":
        repo_name_only = "element-web"
    elif "element-hq" in instance["repo"].lower() and "element-web" in instance["repo"].lower():
        repo_name_only = "element"
        if hsh.endswith("-vnan"):
            hsh = hsh[:-5]
    elif hsh.endswith("-vnan"):
        hsh = hsh[:-5]

    tag = f"{repo_base}.{repo_name_only}-{hsh}"
    if len(tag) > 128:
        tag = tag[:128]
    return f"{dockerhub_username}/sweap-images:{tag}"


@dataclass(frozen=True)
class SampleJob:
    instance: dict[str, Any]
    sample_id: int
    resume_reason: str
    rerun: bool

    @property
    def instance_id(self) -> str:
        return self.instance["instance_id"]


class SWEBenchProEvaluation:
    def __init__(
        self,
        working_dir: str,
        model_config: dict[str, Any] | None = None,
        dataset: str = "ptiny",
        swebench_pro_harness_path: str = "./SWE-bench_Pro-os",
        docker_env_config: str = "",
        dockerhub_username: str = "jefzda",
        run_id: str = "simple-agent-pro",
        num_samples: int = 1,
        nproc: int = 4,
        instance_ids: list[str] | None = None,
        max_steps: int = 200,
        docker_platform: str | None = None,
        skip_image_pull: bool = False,
        eval_python: str | None = None,
        task_profile: str = "default",
    ):
        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")
        if nproc <= 0:
            raise ValueError("nproc must be a positive integer.")

        self.dataset_arg = dataset
        self.dataset_name = "ScaleAI/SWE-bench_Pro"
        self.instances = self._load_instances(dataset)
        if instance_ids:
            self.instances = self._filter_instances(instance_ids)
        self.instances_by_id = {instance["instance_id"]: instance for instance in self.instances}

        self.docker_client = from_env(timeout=300)
        self.image_status: dict[str, bool] = {}
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.harness_path = _resolve_eval_path(swebench_pro_harness_path)
        self.scripts_dir = self.harness_path / "run_scripts"
        self.dockerhub_username = dockerhub_username
        self.run_id = run_id
        self.num_samples = num_samples
        self.nproc = nproc
        self.max_steps = max_steps
        self.docker_platform = docker_platform
        self.task_profile = task_profile
        self.eval_python = (
            _resolve_eval_path(eval_python).absolute().as_posix()
            if eval_python
            else sys.executable
        )
        self.summary: dict[str, Any] = {}
        self.pbar = None
        self.model_config = dict(model_config or {})
        self.model_config["max_steps"] = self.max_steps

        if docker_env_config != "":
            with open(docker_env_config, "r") as f:
                self.docker_env_config = json.load(f)
        else:
            self.docker_env_config = {}

        self.status_log_path = self.working_dir / "run_status.jsonl"
        self.summary_path = self.working_dir / "run_summary.json"
        self.raw_samples_path = self.working_dir / "raw_samples.csv"
        self._write_raw_samples_csv()

        if not skip_image_pull:
            self.pull_images()

    def _load_instances(self, dataset: str) -> list[dict[str, Any]]:
        local_splits = {
            "ptiny": "swebench_pro_tiny.json",
            "pro_tiny": "swebench_pro_tiny.json",
            "tiny": "swebench_pro_tiny.json",
        }
        if dataset in local_splits:
            split_path = _resolve_eval_path(local_splits[dataset])
            with open(split_path, "r") as f:
                return [dict(entry) for entry in json.load(f)]
        if dataset in {"public", "SWE-bench_Pro", "swebench_pro"}:
            return [dict(entry) for entry in load_dataset(self.dataset_name, split="test")]
        dataset_path = _resolve_eval_path(dataset)
        if dataset_path.exists():
            with open(dataset_path, "r") as f:
                return [dict(entry) for entry in json.load(f)]
        raise ValueError(f"Unknown SWE-bench Pro dataset or split: {dataset}")

    def _filter_instances(self, instance_ids: list[str]) -> list[dict[str, Any]]:
        requested_ids = _dedupe(instance_ids)
        available = {instance["instance_id"]: instance for instance in self.instances}
        missing = [instance_id for instance_id in requested_ids if instance_id not in available]
        if missing:
            raise ValueError(f"Requested instance IDs are not in {self.dataset_arg}: {', '.join(missing)}")
        return [available[instance_id] for instance_id in requested_ids]

    def _write_raw_samples_csv(self):
        if not self.instances:
            raise ValueError("No selected SWE-bench Pro instances.")
        fieldnames = list(self.instances[0].keys())
        with open(self.raw_samples_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for instance in self.instances:
                writer.writerow(instance)

    def _image_name(self, instance: dict[str, Any]) -> str:
        return _dockerhub_image_uri(instance, self.dockerhub_username)

    def _check_images(self):
        for instance in tqdm(self.instances, desc="Checking Pro image status"):
            image_name = self._image_name(instance)
            try:
                self.docker_client.images.get(image_name)
                self.image_status[instance["instance_id"]] = True
            except ImageNotFound:
                self.image_status[instance["instance_id"]] = False

        try:
            self.docker_client.images.get("ubuntu:22.04")
        except Exception:
            self.docker_client.images.pull("ubuntu:22.04")

    def pull_images(self):
        self._check_images()
        missing = [
            instance
            for instance in self.instances
            if not self.image_status.get(instance["instance_id"], False)
        ]
        print(f"Total number of selected Pro images: {len(self.image_status)}")
        print(f"Number of Pro images to download: {len(missing)}")
        if not missing:
            return

        async def pull_one(instance: dict[str, Any]) -> bool:
            image_name = self._image_name(instance)
            try:
                if self.docker_platform:
                    await asyncio.to_thread(
                        self.docker_client.images.pull,
                        image_name,
                        platform=self.docker_platform,
                    )
                else:
                    await asyncio.to_thread(self.docker_client.images.pull, image_name)
                print(f"Pulled image: {image_name}")
                return True
            except Exception as e:
                print(f"Failed to pull image {image_name}: {e}")
                return False

        async def async_pull_images():
            semaphore = asyncio.Semaphore(min(self.nproc, 8))
            completed = 0
            failed = 0

            async def guarded_pull(instance):
                nonlocal completed, failed
                async with semaphore:
                    ok = await pull_one(instance)
                    completed += 1
                    if not ok:
                        failed += 1
                    self.pbar.set_postfix_str(f"completed={completed} failed={failed}")
                    self.pbar.update(1)
                    self.pbar.refresh()

            self.pbar = tqdm(total=len(missing), desc="Pulling Pro Docker images", unit="image")
            try:
                await asyncio.gather(*[asyncio.create_task(guarded_pull(instance)) for instance in missing])
            finally:
                self.pbar.close()

        asyncio.run(async_pull_images())

    def _problem_statement(self, instance: dict[str, Any]) -> str:
        sections = [instance.get("problem_statement", "")]
        if instance.get("requirements"):
            sections.append("## Requirements\n\n" + str(instance["requirements"]))
        if instance.get("interface"):
            sections.append("## Interface\n\n" + str(instance["interface"]))
        return "\n\n".join(section for section in sections if section)

    def prepare_experiment_container(self, instance: dict[str, Any]):
        instance_dir = self.working_dir / instance["instance_id"]
        instance_dir.mkdir(parents=True, exist_ok=True)
        with open(instance_dir / "problem_statement.txt", "w") as f:
            f.write(self._problem_statement(instance))

        image_name = self._image_name(instance)
        container = self.docker_client.containers.run(
            image_name,
            entrypoint="/bin/bash",
            command=["-lc", "sleep infinity"],
            detach=True,
            tty=True,
            stdin_open=True,
            working_dir="/app",
            environment=self.docker_env_config.get("experiment_env", None),
            platform=self.docker_platform,
        )

        commands = [
            "git config --global --add safe.directory /app",
            f"cd /app && git reset --hard {instance['base_commit']} && git clean -fd && git checkout {instance['base_commit']}",
        ]
        for command in commands:
            return_code, output = docker_exec(container, f'/bin/bash -lc "{command}"')
            if return_code is not None and return_code != 0:
                print(f"Docker setup command failed for {instance['instance_id']}: {command}")
                print(output)
        return container

    def _patch_name(self, instance_id: str, sample_id: int) -> str:
        return f"{instance_id}__{sample_id}.patch" if sample_id > 0 else f"{instance_id}.patch"

    def _trajectory_name(self, instance_id: str, sample_id: int) -> str:
        return f"{instance_id}__{sample_id}.json" if sample_id > 0 else f"{instance_id}.json"

    def _patch_path(self, instance_id: str, sample_id: int) -> Path:
        return self.working_dir / instance_id / self._patch_name(instance_id, sample_id)

    def _trajectory_path(self, instance_id: str, sample_id: int) -> Path:
        return self.working_dir / instance_id / self._trajectory_name(instance_id, sample_id)

    def _job_state(self, instance_id: str, sample_id: int) -> tuple[str, bool]:
        traj_path = self._trajectory_path(instance_id, sample_id)
        patch_path = self._patch_path(instance_id, sample_id)
        if not traj_path.exists():
            return "missing", False
        try:
            with open(traj_path, "r") as f:
                traj_data = json.load(f)
        except Exception:
            return "corrupted", True
        if traj_data is None or traj_data.get("final_result") is None:
            return "incomplete", True
        if not patch_path.exists():
            return "missing_patch", True
        if patch_path.read_text(errors="replace").strip() == "":
            return "empty_patch", True
        return "complete", False

    def _plan_jobs(self) -> tuple[list[SampleJob], list[SampleJob]]:
        runnable = []
        skipped = []
        for instance in self.instances:
            for sample_id in range(self.num_samples):
                state, rerun = self._job_state(instance["instance_id"], sample_id)
                job = SampleJob(instance=instance, sample_id=sample_id, resume_reason=state, rerun=rerun)
                if state == "complete":
                    skipped.append(job)
                else:
                    runnable.append(job)
        return runnable, skipped

    def _reset_progress_files(self):
        self.status_log_path.write_text("")

    def _initialize_summary(self, runnable: list[SampleJob], skipped: list[SampleJob]):
        planned = len(runnable) + len(skipped)
        self.summary = {
            "dataset_arg": self.dataset_arg,
            "dataset_name": self.dataset_name,
            "run_id": self.run_id,
            "working_dir": self.working_dir.as_posix(),
            "selected_instance_count": len(self.instances),
            "num_samples": self.num_samples,
            "nproc": self.nproc,
            "task_profile": self.task_profile,
            "started_at": _utc_now(),
            "updated_at": _utc_now(),
            "counts": {
                "planned": planned,
                "skipped": len(skipped),
                "runnable": len(runnable),
                "running": 0,
                "completed": 0,
                "failed": 0,
                "rerun": sum(1 for job in runnable if job.rerun),
            },
        }
        self._write_summary()

    def _write_summary(self):
        self.summary["updated_at"] = _utc_now()
        tmp_path = self.summary_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(self.summary, f, indent=2)
            f.write("\n")
        tmp_path.replace(self.summary_path)

    def _record_status(self, record: dict[str, Any]):
        with open(self.status_log_path, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def _progress_postfix(self) -> str:
        counts = self.summary["counts"]
        return (
            f"running={counts['running']} completed={counts['completed']} "
            f"failed={counts['failed']} skipped={counts['skipped']} rerun={counts['rerun']}"
        )

    def _print_startup_summary(self, runnable: list[SampleJob], skipped: list[SampleJob]):
        total_jobs = len(runnable) + len(skipped)
        print("SWE-bench Pro run summary")
        print(f"  dataset arg: {self.dataset_arg}")
        print(f"  dataset name: {self.dataset_name}")
        print(f"  selected instances: {len(self.instances)}")
        print(f"  samples per instance: {self.num_samples}")
        print(f"  total planned jobs: {total_jobs}")
        print(f"  skipped jobs: {len(skipped)}")
        print(f"  runnable jobs: {len(runnable)}")
        print(f"  rerun jobs: {sum(1 for job in runnable if job.rerun)}")
        print(f"  nproc: {self.nproc}")
        print(f"  task profile: {self.task_profile}")
        print(f"  working dir: {self.working_dir}")
        print(f"  harness path: {self.harness_path}")

    def _save_result(self, job: SampleJob, result: AgentResult):
        patch_path = self._patch_path(job.instance_id, job.sample_id)
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text(result.diff)

    async def _run_job(self, job: SampleJob, semaphore: asyncio.Semaphore, pbar: tqdm):
        async with semaphore:
            counts = self.summary["counts"]
            counts["running"] += 1
            pbar.set_postfix_str(self._progress_postfix())
            self._write_summary()

            started_at = _utc_now()
            start_time = time.monotonic()
            return_code = None
            output = None
            error_summary = None
            status = "completed"
            container = None
            print(f"Running Pro instance {job.instance_id}__{job.sample_id}")

            try:
                container = await asyncio.to_thread(self.prepare_experiment_container, job.instance)

                def run_one():
                    agent = CodingAgent(
                        container,
                        repo_path="/app",
                        task_profile=self.task_profile,
                        **self.model_config,
                    )
                    result = agent.run(
                        self._problem_statement(job.instance),
                        trajectory_path=self._trajectory_path(job.instance_id, job.sample_id).as_posix(),
                    )
                    self._save_result(job, result)
                    return_code = 0 if result.status != "fail" else 1
                    return return_code, result.status

                return_code, output = await asyncio.to_thread(run_one)
                if return_code is not None and return_code != 0:
                    status = "failed"
                    error_summary = _tail(output)
                    print(f"Agent failed for {job.instance_id}__{job.sample_id}. Status: {output}")
                else:
                    final_state, _ = self._job_state(job.instance_id, job.sample_id)
                    if final_state != "complete":
                        status = "failed"
                        error_summary = f"Command returned 0 but job state is {final_state}."
            except Exception:
                status = "failed"
                error_summary = _tail(traceback.format_exc())
                print(f"Pro job {job.instance_id}__{job.sample_id} failed.")
                print(traceback.format_exc())
            finally:
                if container is not None:
                    await asyncio.to_thread(_remove_container, container)

            elapsed = time.monotonic() - start_time
            ended_at = _utc_now()
            counts["running"] -= 1
            counts["completed" if status == "completed" else "failed"] += 1
            self._record_status(
                {
                    "instance_id": job.instance_id,
                    "sample_id": job.sample_id,
                    "status": status,
                    "resume_reason": job.resume_reason,
                    "rerun": job.rerun,
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "elapsed_seconds": round(elapsed, 3),
                    "patch_path": self._patch_path(job.instance_id, job.sample_id).as_posix(),
                    "trajectory_path": self._trajectory_path(job.instance_id, job.sample_id).as_posix(),
                    "return_code": return_code,
                    "error_summary": error_summary,
                }
            )
            pbar.update(1)
            pbar.set_postfix_str(self._progress_postfix())
            self._write_summary()

    def run_all(self):
        runnable, skipped = self._plan_jobs()
        self._reset_progress_files()
        self._initialize_summary(runnable, skipped)
        self._print_startup_summary(runnable, skipped)

        for job in skipped:
            self._record_status(
                {
                    "instance_id": job.instance_id,
                    "sample_id": job.sample_id,
                    "status": "skipped",
                    "resume_reason": job.resume_reason,
                    "rerun": False,
                    "started_at": None,
                    "ended_at": _utc_now(),
                    "elapsed_seconds": 0,
                    "patch_path": self._patch_path(job.instance_id, job.sample_id).as_posix(),
                    "trajectory_path": self._trajectory_path(job.instance_id, job.sample_id).as_posix(),
                    "return_code": None,
                    "error_summary": None,
                }
            )

        if not runnable:
            print("No runnable jobs after resume check.")
            self._write_summary()
            return

        async def _run_all_instances():
            semaphore = asyncio.Semaphore(self.nproc)
            pbar = tqdm(total=len(runnable), desc="Running Pro instances", unit="job")
            pbar.set_postfix_str(self._progress_postfix())
            tasks = [asyncio.create_task(self._run_job(job, semaphore, pbar)) for job in runnable]
            try:
                await asyncio.gather(*tasks)
            finally:
                pbar.close()

        asyncio.run(_run_all_instances())

    def get_all_preds(self, instance_ids: list[str] | None = None, sample_id: int = 0) -> Path:
        if not instance_ids:
            instance_ids = [instance["instance_id"] for instance in self.instances]
        preds = []
        for instance_id in instance_ids:
            patch_path = self._patch_path(instance_id, sample_id)
            if not patch_path.exists():
                continue
            patch = patch_path.read_text(errors="replace")
            if patch.strip() == "":
                continue
            preds.append(
                {
                    "instance_id": instance_id,
                    "patch": patch,
                    "prefix": self.run_id if sample_id == 0 else f"{self.run_id}_{sample_id}",
                }
            )
        preds_path = self.working_dir / ("predictions.json" if sample_id == 0 else f"predictions_{sample_id}.json")
        with open(preds_path, "w") as f:
            json.dump(preds, f, indent=2)
            f.write("\n")
        return preds_path

    def run_eval(self, sample_id: int = 0, use_local_docker: bool = True, block_network: bool = False):
        preds_path = self.get_all_preds(sample_id=sample_id)
        output_dir = self.working_dir / ("eval" if sample_id == 0 else f"eval_{sample_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(preds_path, "r") as f:
            preds = json.load(f)
        if not preds:
            result_path = output_dir / "eval_results.json"
            result_path.write_text("{}\n")
            print(f"No non-empty predictions for sample {sample_id}; wrote empty eval results to {result_path}")
            return
        cmd = [
            self.eval_python,
            "swe_bench_pro_eval.py",
            "--raw_sample_path",
            self.raw_samples_path.absolute().as_posix(),
            "--patch_path",
            preds_path.absolute().as_posix(),
            "--output_dir",
            output_dir.absolute().as_posix(),
            "--scripts_dir",
            self.scripts_dir.absolute().as_posix(),
            "--num_workers",
            str(self.nproc),
            "--dockerhub_username",
            self.dockerhub_username,
        ]
        if use_local_docker:
            cmd.append("--use_local_docker")
        if block_network:
            cmd.append("--block_network")
        if self.docker_platform:
            cmd.extend(["--docker_platform", self.docker_platform])
        print(f"Running Pro evaluation with command: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, cwd=self.harness_path.as_posix())
        print(process.stdout.decode(errors="replace"))
        print(process.stderr.decode(errors="replace"))
        if process.returncode != 0:
            raise RuntimeError(f"SWE-bench Pro evaluation failed with return code {process.returncode}")


def _load_instance_ids(args) -> list[str] | None:
    instance_ids: list[str] = []
    if args.instance_ids:
        instance_ids.extend(args.instance_ids)
    if args.instance_ids_file:
        with open(args.instance_ids_file, "r") as f:
            for line in f:
                line = line.strip()
                if line == "" or line.startswith("#"):
                    continue
                instance_ids.append(line)
    if not instance_ids:
        return None
    return _dedupe(instance_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ptiny")
    parser.add_argument("--working-dir", type=str, default="./simple-agent-swebench-pro")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--agent-config", type=str, default=None)
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--instance_ids", nargs="+", type=str)
    parser.add_argument("--instance-ids-file", type=str, default=None)
    parser.add_argument("--swebench-pro-harness-path", type=str, default="../../trae-agent/evaluation/SWE-bench_Pro-os")
    parser.add_argument("--docker-env-config", type=str, default="")
    parser.add_argument("--dockerhub-username", type=str, default="jefzda")
    parser.add_argument("--docker-platform", type=str, default=None)
    parser.add_argument("--eval-python", type=str, default=None)
    parser.add_argument("--run-id", type=str, default="simple-agent-pro")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--task-profile",
        choices=["default", "swebench_pro"],
        default="default",
        help="Optional Pro-only system prompt profile appended at runtime.",
    )
    parser.add_argument("--skip-image-pull", action="store_true")
    parser.add_argument("--block-network", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["e2e", "expr", "eval"],
        default="e2e",
        help="e2e: generate patches and evaluate, expr: only generate patches, eval: only evaluate existing patches",
    )
    args = parser.parse_args()

    if args.config:
        model_config = load_config(args.config)
    elif args.agent_config:
        if not args.provider:
            raise ValueError("--provider is required when --agent-config is used")
        model_config = load_agent_config(args.agent_config, args.provider)
    else:
        model_config = {}

    selected_instance_ids = _load_instance_ids(args)
    evaluation = SWEBenchProEvaluation(
        working_dir=args.working_dir,
        model_config=model_config,
        dataset=args.dataset,
        swebench_pro_harness_path=args.swebench_pro_harness_path,
        docker_env_config=args.docker_env_config,
        dockerhub_username=args.dockerhub_username,
        run_id=args.run_id,
        num_samples=args.num_samples,
        nproc=args.nproc,
        instance_ids=selected_instance_ids,
        max_steps=args.max_steps,
        docker_platform=args.docker_platform,
        skip_image_pull=args.skip_image_pull or args.mode == "eval",
        eval_python=args.eval_python,
        task_profile=args.task_profile,
    )

    if args.mode in {"e2e", "expr"}:
        evaluation.run_all()

    if args.mode in {"e2e", "eval"}:
        for sample_id in range(args.num_samples):
            evaluation.run_eval(sample_id=sample_id, use_local_docker=True, block_network=args.block_network)


if __name__ == "__main__":
    main()
