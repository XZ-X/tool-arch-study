import argparse
import json
import subprocess
import sys
import threading
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

from simple_agent import CodingAgent, AgentResult, load_agent_config, load_config

logger = logging.getLogger(__name__)


def _remove_container(container):
    try:
        container.remove(force=True)
    except Exception:
        pass


def check_existing_result(query_entry) -> tuple[str, int, bool]:
    instance_id = query_entry["instance_id"]
    sample_id = query_entry["sample_id"]
    working_dir = query_entry["working_dir"]

    if sample_id > 0:
        result_fname = f"{instance_id}__{sample_id}.json"
    else:
        result_fname = f"{instance_id}.json"

    result_path = os.path.join(working_dir, instance_id, result_fname)
    if os.path.exists(result_path):
        try:
            with open(result_path, "r") as f:
                result_data = json.load(f)
        except Exception:
            logger.info(
                f"Instance {instance_id}__{sample_id} has a corrupted trajectory. Re-running."
            )
            return (instance_id, sample_id, False)

        if result_data is not None and result_data.get("final_result") is not None:
            logger.info(
                f"Instance {instance_id}__{sample_id} already has a trajectory. Skipping."
            )
            return (instance_id, sample_id, True)
        else:
            logger.info(
                f"Instance {instance_id}__{sample_id} has an incomplete trajectory. Re-running."
            )
            return (instance_id, sample_id, False)
    else:
        return (instance_id, sample_id, False)


class SWEBenchEvaluation:
    def __init__(
        self,
        working_dir: str,
        split: str,
        docker_env_config: str = "",
        swebench_live_harness_path: str = "",
        run_id="coding-agent",
        num_samples: int = 1,
        nproc: int = 4,
        model_config: dict[str, str] | None = None,
    ):
        """
        Initialize the SWEBenchEvaluation class for SWE-bench-Live.

        Args:
            working_dir: The working directory for storing results.
            split: The dataset split to evaluate.
            docker_env_config: The path to the docker environment config file.
            swebench_live_harness_path: The path to the SWE-bench-Live harness.
            run_id: The run id.
            num_samples: Number of samples per instance.
            nproc: Number of parallel processes.
            model_config: Optional dict with 'model' and 'base_url' keys.
        """
        assert split in [
            "lite",
            "verified",
            "vmini",
            "vmini_train",
            "vtiny",
            "test1",
            "test2",
            "test4",
        ], f"Invalid split name: {split}"

        if split == "vmini":
            verified_all_ds = load_dataset(
                "SWE-bench-Live/SWE-bench-Live", split="verified"
            )
            sampled = json.load(open("swebench_live_verified_mini.json", "r"))
            sampled_instance_ids = set([item["instance_id"] for item in sampled])
            self.dataset = verified_all_ds.filter(
                lambda x: x["instance_id"] in sampled_instance_ids
            )
            self.split_name = "verified"
        elif split == "vmini_train":
            verified_all_ds = load_dataset(
                "SWE-bench-Live/SWE-bench-Live", split="verified"
            )
            sampled = json.load(open("swebench_live_verified_mini_train.json", "r"))
            sampled_instance_ids = set([item["instance_id"] for item in sampled])
            self.dataset = verified_all_ds.filter(
                lambda x: x["instance_id"] in sampled_instance_ids
            )
            self.split_name = "verified"
        elif split == "vtiny":
            verified_all_ds = load_dataset(
                "SWE-bench-Live/SWE-bench-Live", split="verified"
            )
            sampled = json.load(open("swebench_live_verified_tiny.json", "r"))
            sampled_instance_ids = set([item["instance_id"] for item in sampled])
            self.dataset = verified_all_ds.filter(
                lambda x: x["instance_id"] in sampled_instance_ids
            )
            self.split_name = "verified"
        elif split in ("test1", "test2", "test4"):
            verified_all_ds = load_dataset(
                "SWE-bench-Live/SWE-bench-Live", split="verified"
            )
            sampled = json.load(open(f"swebench_live_verified_{split}.json", "r"))
            sampled_instance_ids = set([item["instance_id"] for item in sampled])
            self.dataset = verified_all_ds.filter(
                lambda x: x["instance_id"] in sampled_instance_ids
            )
            self.split_name = "verified"
        else:
            self.dataset = load_dataset("SWE-bench-Live/SWE-bench-Live", split=split)
            self.split_name = split

        self.docker_client = from_env()
        self.image_status: dict[Any, Any] = {}
        self.working_dir = Path(working_dir)
        self.swebench_live_harness_path = swebench_live_harness_path
        self.run_id = run_id
        self.model_config = model_config or {}

        if docker_env_config != "":
            with open(docker_env_config, "r") as f:
                self.docker_env_config = json.load(f)
        else:
            self.docker_env_config = {}

        if not self.working_dir.exists():
            self.working_dir.mkdir(parents=True, exist_ok=True)

        self.pull_images()
        self.on_going = 0
        self.pbar = None
        self.num_samples = num_samples
        if nproc <= 0:
            raise ValueError("nproc must be a positive integer.")
        self.nproc = nproc

    def _image_name(self, instance_id: str) -> str:
        key = f"starryzhang/sweb.eval.x86_64.{instance_id.lower()}:latest"
        key = key.replace("__", "_1776_")
        return key

    def _check_images(self):
        for item in tqdm(self.dataset, desc="Checking image status"):
            instance_id = item["instance_id"]
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

        pull_img_on_going = 0

        async def pull_one(instance_id):
            nonlocal pull_img_on_going
            image_name = self._image_name(instance_id)
            try:
                await asyncio.to_thread(self.docker_client.images.pull, image_name)
                print(f"Pulled image: {image_name}")
            except Exception as e:
                print(f"Failed to pull image {image_name}: {e}")
            finally:
                pull_img_on_going -= 1
                self.pbar.set_postfix_str("On-going: %d" % pull_img_on_going)
                self.pbar.update(1)
                self.pbar.refresh()

        async def async_pull_images(instance_ids):
            nonlocal pull_img_on_going
            self.pbar = tqdm(
                total=len(instance_ids),
                desc="Pulling Docker images",
                unit="image",
            )
            tasks = []
            for instance_id in instance_ids:
                pull_img_on_going += 1
                tasks.append(asyncio.create_task(pull_one(instance_id)))
                while pull_img_on_going > 50:
                    await asyncio.sleep(1)
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
            container = self.prepare_container(instance)
            instance_dir = self.working_dir / instance_id
            prefix = self._instance_prefix(instance_id, sample_id)
            trajectory_path = str(instance_dir / f"{prefix}.json")
            try:
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
                threading.Thread(
                    target=lambda c=container: _remove_container(c), daemon=True
                ).start()

    async def async_run_one_instance(self, instance_id, sample_id):
        """
        Run a single instance asynchronously.

        Args:
            instance_id: The ID of the instance to run.
            sample_id: The sample ID.
        """
        instance = None
        for inst in self.dataset:
            if inst["instance_id"] == instance_id:
                instance = inst
        if instance is None:
            print(f"Instance {instance_id} not found.")
            return

        def run_one():
            container = self.prepare_container(instance)
            instance_dir = self.working_dir / instance_id
            prefix = self._instance_prefix(instance_id, sample_id)
            trajectory_path = str(instance_dir / f"{prefix}.json")
            try:
                agent = CodingAgent(container, **self.model_config)
                result = agent.run(instance["problem_statement"], trajectory_path=trajectory_path)
                self._save_result(instance_id, sample_id, result)
                print(
                    f"Instance {instance_id}__{sample_id} completed with status: {result.status}"
                )
            except Exception:
                print(f"Instance {instance_id}__{sample_id} failed.")
                print(traceback.format_exc())
            finally:
                threading.Thread(
                    target=lambda c=container: _remove_container(c), daemon=True
                ).start()

        await asyncio.to_thread(run_one)
        self.on_going -= 1
        self.pbar.update(1)
        self.pbar.set_postfix_str("On-going: %d" % self.on_going)
        self.pbar.refresh()

    def run_all(self):
        """
        Run all instances in the dataset.
        """
        instances = [i for i in self.dataset]
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
        for instance_id, sample_id, skip in should_skip_rets:
            if skip:
                should_skip.add((instance_id, sample_id))
        print(f"Skipping {len(should_skip)} already completed instances.")

        async def _run_all_instances():
            loop = asyncio.get_running_loop()
            loop.set_default_executor(
                ThreadPoolExecutor(max_workers=min(64, self.nproc))
            )
            self.pbar = tqdm(
                total=len(instances) * self.num_samples - len(should_skip),
                desc="Running all instances",
                unit="instance",
            )
            tasks = []
            for instance in instances:
                for sample_id in range(self.num_samples):
                    if (instance["instance_id"], sample_id) in should_skip:
                        continue
                    self.on_going += 1
                    tasks.append(
                        asyncio.create_task(
                            self.async_run_one_instance(
                                instance["instance_id"], sample_id
                            )
                        )
                    )
                    while self.on_going > self.nproc:
                        await asyncio.sleep(1)
            await asyncio.gather(*tasks)

        asyncio.run(_run_all_instances())

    def run_eval(self, sample_id=0):
        """
        Run evaluation using the SWE-bench-Live harness.
        """
        swebench_live_harness_path = Path(self.swebench_live_harness_path)
        swebench_python_path = "swe_live_venv/bin/python3"
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
            "SWE-bench-Live/SWE-bench-Live",
            "--split",
            self.split_name,
            "--namespace",
            "starryzhang",
            "--predictions_path",
            (self.working_dir / suffix).absolute().as_posix(),
            "--max_workers",
            "16",
            "--run_id",
            run_id,
            "--cache_level",
            "instance",
            "--instance_image_tag",
            "latest",
        ]
        print(
            f"Running evaluation with command: {' '.join(cmd)} at {swebench_live_harness_path.as_posix()}"
        )
        process = subprocess.run(
            cmd, capture_output=True, cwd=swebench_live_harness_path.as_posix()
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
    argument_parser.add_argument("--split", type=str, default="vmini")
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
        "--swebench-live-harness-path",
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
        "--config",
        type=str,
        required=False,
        default=None,
        help="Path to single JSON config file (new style).",
    )
    argument_parser.add_argument(
        "--agent-config",
        type=str,
        required=False,
        default=None,
        help="Path to agent_configs.yaml file (old style).",
    )
    argument_parser.add_argument(
        "--provider",
        type=str,
        required=False,
        default=None,
        help="Agent config name to use from the config file (old style).",
    )

    args = argument_parser.parse_args()

    # Support both old and new config styles
    if args.config:
        model_config = load_config(args.config)
    elif args.agent_config:
        model_config = load_agent_config(args.agent_config, args.provider)
    else:
        model_config = {}

    evaluation = SWEBenchEvaluation(
        args.working_dir,
        args.split,
        args.docker_env_config,
        args.swebench_live_harness_path,
        args.run_id,
        args.num_samples,
        args.nproc,
        model_config=model_config,
    )

    if args.mode in ("e2e", "expr"):
        if args.instance_ids:
            print(f"Running instances: {args.instance_ids}")
            for instance_id in tqdm(args.instance_ids, desc="Running instances"):
                evaluation.run_one_instance(instance_id)
        else:
            print("Running all instances")
            evaluation.run_all()

    if args.mode in ("e2e", "eval"):
        for sample_id in range(args.num_samples):
            evaluation.get_all_preds(args.instance_ids, sample_id=sample_id)
            evaluation.run_eval(sample_id=sample_id)


if __name__ == "__main__":
    main()
