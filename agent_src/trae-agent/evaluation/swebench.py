# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import argparse
import asyncio
import json
import shutil
import subprocess
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


def docker_exec(container, command: str):
    """
    Execute a command in a docker container.

    Args:
        container: The docker container object.
        command: The command to execute.

    Returns:
        A tuple of (return_code, output).
    """
    exec_result = container.exec_run(cmd=command)
    return_code = exec_result[0]
    output = exec_result[1].decode("utf-8")
    return return_code, output


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tail(text: str | None, max_chars: int = 2000) -> str | None:
    if text is None:
        return None
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


@dataclass(frozen=True)
class SampleJob:
    instance: dict[str, Any]
    sample_id: int
    resume_reason: str
    rerun: bool

    @property
    def instance_id(self) -> str:
        return self.instance["instance_id"]


class SWEBenchEvaluation:
    def __init__(
        self,
        working_dir: str,
        trae_config_file_name: str,
        dataset: str = "SWE-bench_Verified",
        docker_env_config: str = "",
        swebench_harness_path: str = "",
        run_id="trae-agent",
        num_samples: int = 1,
        nproc: int = 4,
        instance_ids: list[str] | None = None,
    ):
        """
        Initialize the SWEBenchEvaluation class. The initialisation includes checking the existence of required Docker images and downloading missing images.

        Args:
            working_dir: The working directory.
            trae_config_file_name: The path to the Trae config file.
            dataset: The dataset to evaluate.
            docker_env_config: The path to the docker environment config file.
            swebench_harness_path: The path to the SWEBench harness.
            run_id: The run id.
            num_samples: Number of samples per instance.
            nproc: Maximum number of concurrent instance/sample jobs.
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
        ], (
            f"Invalid dataset name: {dataset}"
        )
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
        }
        if dataset in verified_local_splits:
            verified_all_ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
            split_file = verified_local_splits[dataset]
            sampled = json.load(open(split_file, "r"))
            sampled_instance_ids = {item["instance_id"] for item in sampled}
            self.dataset = verified_all_ds.filter(
                lambda x: x["instance_id"] in sampled_instance_ids
            )
            self.dataset_name = "SWE-bench_Verified"
        else:
            self.dataset = load_dataset(f"princeton-nlp/{dataset}", split="test")
            self.dataset_name = dataset

        if instance_ids:
            self.dataset = self._filter_dataset_to_instances(instance_ids)

        self.instances = [dict(instance) for instance in self.dataset]
        self.instances_by_id = {instance["instance_id"]: instance for instance in self.instances}

        self.docker_client = from_env(timeout=300)
        self.image_status: dict[Any, Any] = {}
        self.working_dir = Path(working_dir)
        self.swebench_harness_path = swebench_harness_path
        self.run_id = run_id

        if docker_env_config != "":
            with open(docker_env_config, "r") as f:
                self.docker_env_config = json.load(f)
        else:
            self.docker_env_config = {}

        self.working_dir.mkdir(parents=True, exist_ok=True)

        self.trae_config_file_name = trae_config_file_name

        shutil.copyfile(
            self.trae_config_file_name, self.working_dir / "trae_config_local.json"
        )

        self.status_log_path = self.working_dir / "run_status.jsonl"
        self.summary_path = self.working_dir / "run_summary.json"
        self.num_samples = num_samples
        self.nproc = nproc
        self.summary: dict[str, Any] = {}
        self.pbar = None

        self.pull_images()

    def _filter_dataset_to_instances(self, instance_ids: list[str]):
        requested_ids = _dedupe(instance_ids)
        available_ids = {instance["instance_id"] for instance in self.dataset}
        missing_ids = [instance_id for instance_id in requested_ids if instance_id not in available_ids]
        if missing_ids:
            missing = ", ".join(missing_ids)
            raise ValueError(f"Requested instance IDs are not in {self.dataset_arg}: {missing}")

        requested_id_set = set(requested_ids)
        return self.dataset.filter(lambda x: x["instance_id"] in requested_id_set)

    def _image_name(self, instance_id: str) -> str:
        """
        Get the image name from the instance id.

        Args:
            instance_id: The instance id.

        Returns:
            The image name.
        """
        key = f"swebench/sweb.eval.x86_64.{instance_id.lower()}:latest"
        key = key.replace("__", "_1776_")
        return key

    def _check_images(self):
        """
        Check the existence of required Docker images.
        """
        for item in tqdm(self.instances, desc="Checking image status"):
            instance_id = item["instance_id"]
            image_name = self._image_name(instance_id)
            try:
                _ = self.docker_client.images.get(image_name)
                self.image_status[instance_id] = True
            except ImageNotFound:
                self.image_status[instance_id] = False
        try:
            _ = self.docker_client.images.get("ubuntu:22.04")
        except Exception:
            self.docker_client.images.pull("ubuntu:22.04")

    def pull_images(self):
        """
        Pull the required Docker images.
        """
        self._check_images()
        print(f"Total number of selected images: {len(self.image_status)}")
        instance_ids = [
            instance_id
            for instance_id in self.image_status
            if not self.image_status[instance_id]
        ]
        print(f"Number of images to download: {len(instance_ids)}")
        if len(instance_ids) == 0:
            return

        async def pull_one(instance_id) -> bool:
            """
            Asynchronously pull a single Docker image.
            """
            image_name = self._image_name(instance_id)
            try:
                await asyncio.to_thread(self.docker_client.images.pull, image_name)
                print(f"Pulled image: {image_name}")
                return True
            except Exception as e:
                print(f"Failed to pull image {image_name}: {e}")
                return False

        async def async_pull_images(instance_ids):
            """
            Asynchronously pull Docker images for the given instance IDs.
            """
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

    def prepare_trae_agent(self):
        """
        Prepare the Trae agent by building Trae Agent and UV inside a general Ubuntu image, save the artifacts in the workspace, which are then used in experiment Docker containers.
        """
        tars = ["trae-agent.tar", "uv.tar", "uv_shared.tar"]
        all_exist = True
        for tar in tars:
            tar_path = self.working_dir / tar
            if not tar_path.exists():
                all_exist = False
                break

        if all_exist:
            print("Found built trae-agent and uv artifacts. Skipping building.")
            return

        try:
            image = self.docker_client.images.get("ubuntu:22.04")
        except Exception:
            image = self.docker_client.images.pull("ubuntu:22.04")

        container = self.docker_client.containers.run(
            image,
            command="bash",
            detach=True,
            tty=True,
            stdin_open=True,
            volumes={
                self.working_dir.absolute(): {"bind": "/trae-workspace", "mode": "rw"}
            },
            # ports={'2333/tcp': 2333},
            # extra_hosts={"host.docker.internal": "host-gateway"},
            environment=self.docker_env_config.get("preparation_env", None),
        )

        commands = [
            "apt-get update",
            "apt-get install -y curl git",
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            "export UV_LINK_MODE=copy; cd /trae-workspace/trae-agent && source $HOME/.local/bin/env && ( uv sync || uv sync )",
            # "export UV_LINK_MODE=copy; cd /trae-workspace/trae-agent && source $HOME/.local/bin/env && ( uv sync || uv sync )",
        ]

        for command in tqdm(
            commands, desc="Building trae-agent inside base Docker container"
        ):
            try:
                new_command = f'/bin/bash -c "{command}"'
                return_code, output = docker_exec(container, new_command)
            except Exception:
                print(f"{command} failed.")
                print(traceback.format_exc())
                break
            if return_code is not None and return_code != 0:
                print("Docker exec error. Error message: {}".format(output))
                exit(-1)

        with open(self.working_dir / "trae-agent.tar", "wb") as f:
            bits, _ = container.get_archive("/trae-workspace/trae-agent")
            for chunk in bits:
                f.write(chunk)

        with open(self.working_dir / "uv.tar", "wb") as f:
            bits, _ = container.get_archive("/root/.local/bin/uv")
            for chunk in bits:
                f.write(chunk)

        with open(self.working_dir / "uv_shared.tar", "wb") as f:
            bits, _ = container.get_archive("/root/.local/share/uv")
            for chunk in bits:
                f.write(chunk)

        container.stop()
        container.remove()

    def prepare_experiment_container(self, instance):
        """
        Prepare an experiment Docker container for a given instance.

        Args:
            instance: A dictionary containing instance information.

        Returns:
            The Docker container object.
        """
        image_name = self._image_name(instance["instance_id"])

        instance_dir = self.working_dir / instance["instance_id"]
        instance_dir.mkdir(parents=True, exist_ok=True)

        with open(instance_dir / "problem_statement.txt", "w") as f:
            f.write(instance["problem_statement"])

        container = self.docker_client.containers.run(
            image_name,
            command="/bin/bash",
            detach=True,
            tty=True,
            stdin_open=True,
            volumes={
                self.working_dir.absolute(): {"bind": "/trae-workspace", "mode": "rw"}
            },
            # ports={'2333/tcp': 2333},
            # network="host",
            # extra_hosts={"host.docker.internal": "host-gateway"},
            working_dir="/trae-workspace",
            environment=self.docker_env_config.get("experiment_env", None),
            stream=True,
        )

        commands = [
            # "tar xf trae-agent.tar -C /root/",
            "tar xf uv.tar -C /root/",
            "mkdir -p /root/.local/bin",
            "mv /root/uv /root/.local/bin/",
            "tar xf uv_shared.tar -C /root/",
            "mkdir -p /root/.local/share",
            "mv /root/uv /root/.local/share/",
            "cd /trae-workspace/trae-agent && git config --global --add safe.directory /trae-workspace/trae-agent && git stash",
        ]

        for command in commands:
            try:
                new_command = f'/bin/bash -c "{command}"'
                return_code, output = docker_exec(container, new_command)
                if return_code is not None and return_code != 0:
                    print(
                        "Docker exec error. Returned code {}. Error message: {}".format(
                            return_code, output
                        )
                    )
            except Exception:
                print(f"{command} failed.")
                print(traceback.format_exc())
                break
        return container

    def _patch_name(self, instance_id: str, sample_id: int) -> str:
        if sample_id > 0:
            return f"{instance_id}__{sample_id}.patch"
        return f"{instance_id}.patch"

    def _trajectory_name(self, instance_id: str, sample_id: int) -> str:
        if sample_id > 0:
            return f"{instance_id}__{sample_id}.json"
        return f"{instance_id}.json"

    def _patch_path(self, instance_id: str, sample_id: int) -> Path:
        return self.working_dir / instance_id / self._patch_name(instance_id, sample_id)

    def _trajectory_path(self, instance_id: str, sample_id: int) -> Path:
        return self.working_dir / instance_id / self._trajectory_name(instance_id, sample_id)

    def _trajectory_state(self, instance_id: str, sample_id: int) -> tuple[str, bool]:
        traj_path = self._trajectory_path(instance_id, sample_id)
        if not traj_path.exists():
            return "missing", False
        try:
            with open(traj_path, "r") as f:
                traj_data = json.load(f)
        except Exception:
            return "corrupted", True

        if traj_data is not None and traj_data.get("final_result") is not None:
            patch_path = self._patch_path(instance_id, sample_id)
            if not patch_path.exists():
                return "empty_patch", True
            try:
                if patch_path.read_text().strip() == "":
                    return "empty_patch", True
            except Exception:
                return "empty_patch", True
            return "complete", False
        return "incomplete", True

    def _plan_jobs(self) -> tuple[list[SampleJob], list[SampleJob]]:
        runnable: list[SampleJob] = []
        skipped: list[SampleJob] = []
        for instance in self.instances:
            for sample_id in range(self.num_samples):
                state, rerun = self._trajectory_state(instance["instance_id"], sample_id)
                job = SampleJob(
                    instance=instance,
                    sample_id=sample_id,
                    resume_reason=state,
                    rerun=rerun,
                )
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
            f"running={counts['running']} "
            f"completed={counts['completed']} "
            f"failed={counts['failed']} "
            f"skipped={counts['skipped']} "
            f"rerun={counts['rerun']}"
        )

    def _print_startup_summary(self, runnable: list[SampleJob], skipped: list[SampleJob]):
        total_jobs = len(runnable) + len(skipped)
        print("SWE-bench Verified run summary")
        print(f"  dataset arg: {self.dataset_arg}")
        print(f"  dataset name: {self.dataset_name}")
        print(f"  selected instances: {len(self.instances)}")
        print(f"  samples per instance: {self.num_samples}")
        print(f"  total planned jobs: {total_jobs}")
        print(f"  skipped jobs: {len(skipped)}")
        print(f"  runnable jobs: {len(runnable)}")
        print(f"  rerun jobs: {sum(1 for job in runnable if job.rerun)}")
        print(f"  nproc: {self.nproc}")
        print(f"  working dir: {self.working_dir}")

    def _build_agent_command(self, job: SampleJob) -> str:
        instance_id = job.instance_id
        instance_dir = instance_id
        problem_statement_path = instance_dir + "/problem_statement.txt"
        patch_file_path = instance_dir + "/" + self._patch_name(instance_id, job.sample_id)
        traj_path = instance_dir + "/" + self._trajectory_name(instance_id, job.sample_id)
        command = (
            f"source trae-agent/.venv/bin/activate && "
            f"trae-cli run {problem_statement_path} "
            f'--working-dir="/testbed/" '
            f"--config-file trae_config_local.json "
            f"--problem-instance {instance_id} "
            f"--max-steps 200 "
            f"--must-patch "
            f"--patch-path {patch_file_path} "
            f"--trajectory-file {traj_path}"
        )
        return f"/bin/bash -c '{command}'"

    async def _run_job(
        self,
        job: SampleJob,
        semaphore: asyncio.Semaphore,
        pbar: tqdm,
    ):
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
            command = self._build_agent_command(job)
            print(f"Running instance {job.instance_id}__{job.sample_id}")

            try:
                container = await asyncio.to_thread(self.prepare_experiment_container, job.instance)

                def run_one():
                    try:
                        return docker_exec(container, command)
                    finally:
                        try:
                            container.stop()
                        except Exception:
                            pass

                return_code, output = await asyncio.to_thread(run_one)
                if return_code is not None and return_code != 0:
                    status = "failed"
                    error_summary = _tail(output)
                    print(f"Docker exec error for {job.instance_id}__{job.sample_id}. Error message: {output}")
                else:
                    final_state, _ = self._trajectory_state(job.instance_id, job.sample_id)
                    if final_state != "complete":
                        status = "failed"
                        error_summary = f"Command returned 0 but trajectory state is {final_state}."
            except Exception:
                status = "failed"
                error_summary = _tail(traceback.format_exc())
                print(f"{command} failed.")
                print(traceback.format_exc())
                if container is not None:
                    try:
                        await asyncio.to_thread(container.stop)
                    except Exception:
                        pass

            elapsed = time.monotonic() - start_time
            ended_at = _utc_now()
            counts["running"] -= 1
            if status == "completed":
                counts["completed"] += 1
            else:
                counts["failed"] += 1

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

    def run_one_instance(self, instance_id):
        """
        Run one selected instance through the same resumable scheduler used by run_all.
        """
        if instance_id not in self.instances_by_id:
            print(f"Instance {instance_id} not found.")
            return
        old_instances = self.instances
        old_instances_by_id = self.instances_by_id
        try:
            instance = self.instances_by_id[instance_id]
            self.instances = [instance]
            self.instances_by_id = {instance_id: instance}
            self.run_all()
        finally:
            self.instances = old_instances
            self.instances_by_id = old_instances_by_id

    def run_all(self):
        """
        Run all selected instances in the dataset with bounded parallelism and resume-aware skipping.
        """
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
            pbar = tqdm(
                total=len(runnable),
                desc="Running selected instances",
                unit="job",
            )
            pbar.set_postfix_str(self._progress_postfix())
            tasks = [asyncio.create_task(self._run_job(job, semaphore, pbar)) for job in runnable]
            try:
                await asyncio.gather(*tasks)
            finally:
                pbar.close()

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

        result_filename = f"trae-agent.{self.run_id}.json"
        print(f"Evaluation completed and file saved to {result_filename}")

    def get_all_preds(self, instance_ids: list[str] | None = None, sample_id: int = 0):
        """
        Get all predictions for a list of instance IDs.

        Args:
            instance_ids: A list of instance IDs. If None, all selected instances in the dataset will be used.
        """
        preds = []
        if not instance_ids:
            instance_ids = [instance["instance_id"] for instance in self.instances]
        for instance_id in instance_ids:
            patch_path = self._patch_path(instance_id, sample_id)
            if not patch_path.exists():
                continue
            with open(patch_path, "r") as f:
                patch = f.read()
            preds.append(
                {
                    "instance_id": instance_id,
                    "model_name_or_path": "trae-agent",
                    "model_patch": patch,
                }
            )
        if sample_id > 0:
            preds_path = self.working_dir / f"predictions_{sample_id}.json"
        else:
            preds_path = self.working_dir / "predictions.json"
        with open(preds_path, "w") as f:
            json.dump(preds, f)


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


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
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--dataset", type=str, default="SWE-bench_Verified")
    argument_parser.add_argument("--working-dir", type=str, default="./trae-workspace")
    argument_parser.add_argument(
        "--config-file", type=str, default="trae_config_local.json"
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
        default="trae-agent",
        help="Run ID for SWE-bench evaluation.",
    )
    argument_parser.add_argument(
        "--num_samples",
        type=int,
        required=False,
        default=1,
        help="Number of samples for each run.",
    )
    argument_parser.add_argument(
        "--nproc",
        type=int,
        required=False,
        default=4,
        help="Number of processes to run in parallel.",
    )
    # expr: only generate patches
    # eval: only evaluation patches
    # e2e: both expr and eval
    argument_parser.add_argument(
        "--mode",
        type=str,
        choices=["e2e", "expr", "eval"],
        default="e2e",
        help="e2e: both expr and eval, expr: only generate patches, eval: only evaluation patches",
    )

    args = argument_parser.parse_args()
    selected_instance_ids = _load_instance_ids(args)
    evaluation = SWEBenchEvaluation(
        args.working_dir,
        args.config_file,
        args.dataset,
        args.docker_env_config,
        args.swebench_harness_path,
        args.run_id,
        args.num_samples,
        args.nproc,
        selected_instance_ids,
    )

    if args.mode == "e2e" or args.mode == "expr":
        evaluation.prepare_trae_agent()
        print("Running selected instances")
        evaluation.run_all()

    if args.mode == "e2e" or args.mode == "eval":
        for sample_id in range(args.num_samples):
            evaluation.get_all_preds(selected_instance_ids, sample_id=sample_id)
            evaluation.run_eval(sample_id=sample_id)


if __name__ == "__main__":
    main()
