# editagent_script.py

import openai
import re
import yaml
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import concurrent.futures
import docker

import logging as _logging

from r2egym.agenthub.runtime.docker import DockerRuntime
from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.agent.agent import AgentArgs, Agent, HealthCheckFailureException

from r2egym.docker_bash_utils.docker_list_tags import fetch_docker_tags
from r2egym.agenthub.utils.log import get_logger
from r2egym.logging import setup_logging, INFO
from r2egym.agenthub.utils.utils import get_parsed_commit

from fire import Fire
from r2egym.agenthub.utils.utils import match_dockerimage_to_repo
from r2egym.agenthub import SUPPORTED_REPOS
from datasets import load_dataset
from r2egym.agenthub.trajectory import TrajectoryStep, Trajectory
import time

##############################################################################
# Initialize Logger
##############################################################################
logger = get_logger(__name__)  # Initialize the logger



##############################################################################
# Utility Function
##############################################################################
def get_docker_images(repo_name) -> List[str]:
    """
    Fetches the list of Docker images available for the base image.

    Returns:
        A list of Docker image tags.
    """
    base_image = f"namanjain12/{repo_name}new"
    tags = fetch_docker_tags(base_image)
    docker_image_list = [f"{base_image}:{x['name']}" for x in tags]
    return docker_image_list


def prepull_docker_image(docker_image: str) -> bool:
    """
    Prepulls a single Docker image.
    
    Args:
        docker_image: The Docker image name to pull
        
    Returns:
        True if successful, False otherwise
    """
    try:
        client = docker.from_env()
        logger.info(f"Pulling Docker image: {docker_image}")
        client.images.pull(docker_image)
        logger.info(f"Successfully pulled Docker image: {docker_image}")
        return True
    except Exception as e:
        logger.error(f"Failed to pull Docker image {docker_image}: {e}")
        return False


def prepull_docker_images(ds_selected: List[Dict], max_workers: Optional[int] = None) -> None:
    """
    Prepulls all Docker images in parallel before starting the main execution.
    
    Args:
        ds_selected: List of dataset entries containing docker_image keys
        max_workers: Maximum number of threads for parallel pulling
    """
    # Extract unique Docker images
    docker_images = list(set([ds_entry["docker_image"] for ds_entry in ds_selected]))
    logger.info(f"Starting parallel prepull of {len(docker_images)} unique Docker images...")
    
    # Use ThreadPoolExecutor for I/O bound operations like Docker pulls
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all pull tasks
        future_to_image = {
            executor.submit(prepull_docker_image, docker_image): docker_image
            for docker_image in docker_images
        }
        
        # Track results
        successful_pulls = []
        failed_pulls = []
        
        for future in concurrent.futures.as_completed(future_to_image):
            docker_image = future_to_image[future]
            try:
                success = future.result()
                if success:
                    successful_pulls.append(docker_image)
                else:
                    failed_pulls.append(docker_image)
            except Exception as e:
                logger.error(f"Exception during prepull of {docker_image}: {e}")
                failed_pulls.append(docker_image)
    
    logger.info(f"Prepull completed. Success: {len(successful_pulls)}, Failed: {len(failed_pulls)}")
    if failed_pulls:
        logger.warning(f"Failed to pull images: {failed_pulls}")


##############################################################################
# Docker Image Logger Adapter
##############################################################################
class DockerImageLoggerAdapter(_logging.LoggerAdapter):
    """Logger adapter that prepends docker image name to log messages."""

    # Common prefixes to strip for cleaner logs
    _STRIP_PREFIXES = ("slimshetty/swebench-verified:sweb.eval.x86_64.",)

    def process(self, msg, kwargs):
        image = self.extra["docker_image"]
        for prefix in self._STRIP_PREFIXES:
            if image.startswith(prefix):
                image = image[len(prefix):]
                break
        return f'[{image}] {msg}', kwargs


##############################################################################
# editagent Functions
##############################################################################
def run_agent_with_restarts(
    agent,
    env,
    max_steps=40,
    num_restarts=1,
    temperature=0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_steps_absolute=50,
    use_fn_calling: bool = True,
    max_iterations: int = 1,
    scaffold: str = "r2egym",
    max_tokens: int = 65536,
    max_completion_tokens: Optional[int] = None,
    max_total_time: int = 3000, # 50 mins for one trajactory
    max_llm_time: int = 600, # 10 mins for each llm query time
):
    """
    Iterative eval protocol:
    - normally run the agent
    - run for maximum num_iterations = 3 times
    - stop if trajectory.exit_reason == "agent"
    - otherwise continue iteratively till maximum iterations
    - finally choose the trajectory with the lowest number of steps
    - note restarts and iterative_evals are different (so just use one of them | add an assert flag)
    - also if original is at temp = 0, then we do next with 0.1 and 0.2 and so on (max 0.2)
    """
    steps_per_agent = max_steps // num_restarts
    logger.warning(f"running {steps_per_agent} steps per agent")

    # only one of restarts > 1 and iterative_eval can be True
    iterative_eval = max_iterations > 1
    assert not (num_restarts > 1 and iterative_eval), "only one of restarts > 1 and iterative_eval can be True"
    logger.warning(f"Using iterations: {max_iterations}, using iterative protocol: {iterative_eval}")

    # if original is at temp = 0, then we do next with 0.1 and 0.2 and so on (max 0.2)
    # if temperature is 0, create list of increasing temperatures up to 0.2
    if temperature == 0:
        temperatures = [0.0 + 0.1 * i for i in range(max_iterations)]
        temperatures = [min(t, 0.2) for t in temperatures]  # cap at 0.2
    else:
        temperatures = [temperature] * max_iterations
    logger.warning(f"Using temperatures: {temperatures}")

    # run the agent in iterative protocol
    trajectories = []
    for iteration in range(max_iterations):
        for idx in range(num_restarts):
            logger.warning(f"running agent at idx: {idx+1}")
            trajectory = agent.run(
                env,
                max_steps=steps_per_agent,
                temperature=temperatures[iteration],
                top_p=top_p,
                top_k=top_k,
                max_steps_absolute=max_steps_absolute,
                use_fn_calling=use_fn_calling,
                scaffold=scaffold,
                max_token_limit=max_tokens,
                max_total_time=max_total_time,
                max_llm_time=max_llm_time,
            )
            # remove reproduce.py
            # env.runtime.run('rm reproduce_issue.py')
        if trajectory.exit_reason == "agent":
            logger.warning(f"agent self-finished at iteration: {iteration}")
            return trajectory
        # otherwise continue iteratively
        trajectories.append(trajectory)
        # reset the env
        # env.reset()

    # choose the trajectory with the lowest number of steps
    trajectory = min(trajectories, key=lambda x: x.num_steps)
    return trajectory


def _get_gold_patch(env, ds) -> str:
    """Return the ground-truth solution patch for the current task.

    SWE-bench-schema datasets (swebench_verified / swesmith / swerebench) carry the gold solution in
    ds["patch"]. r2e datasets instead store a parsed commit on the runtime, from which we reconstruct the solution
    using the non-test source files. We keep only_python=True to match the validated reconstruction in
    validate_docker_and_hf.py (which asserts reward == 1 for the gold patch); r2e repos are Python, and including
    non-Python hunks risks an atomic `git apply` rejection.
    """
    patch = ds.get("patch") if hasattr(ds, "get") else None
    if patch:
        return patch
    commit = getattr(env.runtime, "commit", None)
    if commit is not None:
        return commit.get_patch(test_file=False, non_test_file=True, only_python=True)
    raise ValueError(
        "No gold patch available: ds has no 'patch' field and the runtime has no parsed commit."
    )


def runagent(
    ds,
    exp_name: Optional[str] = None,
    max_steps=40,
    num_restarts=1,
    max_steps_absolute=50,
    llm_name="gpt-4o",
    temperature=0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    use_fn_calling: bool = True,
    backend: str = "kubernetes", # "kubernetes" or "docker"
    max_reward_calc_time: int = 300,
    max_iterations: int = 1,
    scaffold: str = "r2egym",
    max_tokens: int = 65536,
    max_completion_tokens: Optional[int] = None,
    max_total_time: int = 3000, # 50 mins for one trajactory
    max_llm_time: int = 600, # 10 mins for each llm query time
    max_llm_retries: Optional[int] = None,
    oracle: bool = False, # if True, skip the agent and apply the gold patch (env sanity check)
) -> Optional[str]:
    """
    Runs the editagent agent on a specified Docker image.

    Args:
        docker_image: The Docker image to use for the environment.
        traj_dir: Directory to save trajectories.
        jsonl_file: Path to the JSONL file to save results. If not provided, generated using traj_dir and exp_name.
        exp_name: Experiment name. Used if jsonl_file is not provided. If not provided, a unique name is generated.
    """
    logger = setup_logging(
        name=ds["docker_image"].replace("/", "_"),
        log_file=f"run_logs/{exp_name}/{ds['docker_image'].replace('/', '_')}.log",
        console=True,
        level=INFO,
    )
    logger = DockerImageLoggerAdapter(logger, {"docker_image": ds["docker_image"]})
    logger.info(f"Starting editagent on Docker image: {ds['docker_image']}")
    logger.info(f"Using LLM: {llm_name}")
    logger.info(f"Max Steps: {max_steps}")

    assert scaffold in ["r2egym", "sweagent", "openhands"], f"Scaffold is {scaffold}, must be one of [r2egym, sweagent, openhands]"
    # Generate a unique experiment name if not provided
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize environment arguments
    env_args = EnvArgs(ds=ds)

    # Initialize the RepoEnv
    env = RepoEnv(env_args, logger=logger, backend=backend)
    # set agent args
    if use_fn_calling:
        assert scaffold != "sweagent", "SWEagent scaffold does not support fn calling"
        cfg_path = Path(
            f"./src/r2egym/agenthub/config/{scaffold}/edit_fn_calling.yaml"
        )
        agent_args = AgentArgs.from_yaml(cfg_path)
    else:
        cfg_path = Path(
            f"./src/r2egym/agenthub/config/{scaffold}/edit_non_fn_calling.yaml"
        )
        agent_args = AgentArgs.from_yaml(cfg_path)
    agent_args.llm_name = llm_name

    # Set max_completion_tokens in other_args if provided
    if max_completion_tokens is not None:
        if agent_args.other_args is None:
            agent_args.other_args = {}
        agent_args.other_args["max_completion_tokens"] = max_completion_tokens
    
    if max_llm_retries is not None:
        if agent_args.other_args is None:
            agent_args.other_args = {}
        agent_args.other_args["max_retries"] = max_llm_retries

    # Initialize the agent
    agent = Agent(name="EditAgent", args=agent_args, logger=logger)

    # run agent editagent
    try:
        if oracle:
            # Oracle / env sanity-check mode: do not run the agent. Apply the ground-truth gold patch directly to the
            # freshly started (buggy-state) container, then fall through to the normal reward computation. A correctly
            # set up environment should grade this as a full pass (reward == 1).
            gold_patch = _get_gold_patch(env, ds)
            apply_out, apply_err = env.runtime.apply_patch(gold_patch)
            # apply_patch does not raise on a failed `git apply`; it returns "0" on success or an "Error: ..." string.
            # Distinguish "gold patch did not apply" from "tests failed" so a non-applying patch is not mistaken for a
            # broken environment.
            apply_ok = str(apply_err).strip() == "0"
            if not apply_ok:
                # Image-build side effects can leave tracked files dirty (e.g. Go's go.work.sum regenerated by
                # `go mod` steps during the build), so a plain `git apply` of the gold patch fails on unrelated
                # context lines. Mirror the official SWE-bench / SWE-bench-Pro harness, which resets the worktree
                # to base_commit before applying the patch, and retry once.
                base_commit = ds.get("base_commit")
                if base_commit:
                    logger.warning(
                        f"[oracle] gold patch did not apply on the as-built tree (exit={apply_err}); "
                        f"resetting to base_commit {base_commit} and retrying."
                    )
                    env.runtime.run(f"git reset --hard {base_commit}")
                    apply_out, apply_err = env.runtime.apply_patch(gold_patch)
                    apply_ok = str(apply_err).strip() == "0"
            if apply_ok:
                logger.warning(f"[oracle] applied gold patch: {apply_out.strip()[:500]}")
            else:
                logger.error(
                    f"[oracle] gold patch FAILED to apply (exit={apply_err}); reward below will reflect the unpatched "
                    f"tree, not an env problem. Output: {apply_out.strip()[:1000]}"
                )
            trajectory = Trajectory(
                trajectory_steps=[],
                problem_statement=env.runtime.get_task_instruction(),
                docker_image=ds["docker_image"],
                env_args=asdict(env_args),
                agent_args=asdict(agent_args),
                max_steps=max_steps,
                max_steps_absolute=max_steps_absolute,
                max_token_limit=max_tokens,
                max_llm_time=max_llm_time,
                max_exec_time=90,
                max_total_time=max_total_time,
                exit_reason="oracle" if apply_ok else "oracle_apply_failed",
                output_patch=gold_patch,
                num_steps_used=0,
            )
        else:
            trajectory = run_agent_with_restarts(
                agent,
                env,
                max_steps=max_steps,
                num_restarts=num_restarts,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_steps_absolute=max_steps_absolute,
                use_fn_calling=use_fn_calling,
                max_iterations=max_iterations,
                scaffold=scaffold,
                max_tokens=max_tokens,
                max_completion_tokens=max_completion_tokens,
                max_total_time=max_total_time,
                max_llm_time=max_llm_time,
            )
    except HealthCheckFailureException as e:
        logger.critical(
            f"Health check failed during agent run for Docker image {ds['docker_image']}: {e}"
        )
        logger.critical("LLM server is unhealthy. Cannot continue.")
        env.close()
        raise  # Re-raise to propagate to runagent_multiple
    except Exception as e:
        logger.error(
            f"Error during agent run for Docker image {ds['docker_image']}: {e}"
        )
        return None

    # also get the gt outputs
    reward_calc_time = time.time()
    reward, test_output = env.runtime._calculate_reward(get_test_output=True, timeout=max_reward_calc_time)
    reward_calc_time = time.time() - reward_calc_time
    logger.warning(f"Reward: {reward}")
    # Close the environment and runtime
    env.close()

    # update the trajectory object
    trajectory.reward = reward
    trajectory.test_output = test_output
    trajectory.ds = ds
    trajectory.exp_name = exp_name
    trajectory.reward_calc_time = reward_calc_time # time taken to calculate reward
    logger.warning(f"time taken to calculate reward in seconds: {reward_calc_time:.2f}")

    logger.info(f"editagent completed for Docker image: {ds['docker_image']}")
    # close env and docker runtime
    logger.info(f"Closing environment for Docker image: {ds['docker_image']}")
    return trajectory.model_dump_json()


def runagent_multiple(
    dataset: str,
    split: str,
    k: int = 1,
    traj_dir: str = "./traj",
    exp_name: Optional[str] = None,
    start_idx=0,
    max_steps=40,
    num_restarts=1,
    max_steps_absolute=50,
    max_workers: Optional[int] = None,
    llm_name="gpt-4o",
    use_existing: bool = True,
    skip_existing: bool = False,
    temperature: float = 0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    use_fn_calling: bool = True,
    backend: str = "kubernetes", # "kubernetes" or "docker"
    max_reward_calc_time: int = 300,
    max_iterations: int = 1,
    scaffold: str = "r2egym",
    prepull_images: bool = False,
    max_tokens: int = 65536,
    max_completion_tokens: Optional[int] = None,
    max_total_time: int = 3000, # 50 mins for one trajactory
    max_llm_time: int = 600, # 10 mins for each llm query time
    max_llm_retries: Optional[int] = None,
    oracle: bool = False, # if True, skip the agent and apply the gold patch (env sanity check)
):
    """
    Runs the editagent agent on the first k Docker images.

    Args:
        k: The number of Docker images to process.
        traj_dir: Directory to save trajectories.
        exp_name: Experiment name for the JSONL file. If not provided, a unique name is generated.
        start_idx: The starting index in the Docker images list.
        max_steps: Maximum steps for the agent run.
        max_workers: Maximum number of threads to use.
        prepull_images: Whether to prepull Docker images in parallel before starting execution.
    """
    logger.info(f"args = {locals()}")
    # Show process id
    import os
    logger.info(f"process id = {os.getpid()}")
    
    # Load the dataset
    ds = load_dataset(dataset, split=split)
    logger.info(f"{len(ds)}, {k}, {start_idx}")
    # shuffle the dataset
    ds = ds.shuffle(seed=42)

    # get selected idxs — clip to dataset size so the last split (which may be shorter than ceil(K/NUM_SPLITS) when
    # K is not a multiple) doesn't crash with IndexError.
    end_idx = min(start_idx + k, len(ds))
    selected_idx = range(start_idx, end_idx)
    if end_idx <= start_idx:
        logger.warning(
            f"start_idx={start_idx} is at or past dataset end (size={len(ds)}); nothing to do."
        )
        return
    raw_selected = [ds[i] for i in selected_idx]

    ds_selected = [r for r in raw_selected if r.get("docker_image")]

    # print ds_selected stats
    logger.info(
        f"Dataset: {dataset}, Split: {split}, Num_total: {len(ds)}, Start Index: {start_idx}, k: {k}"
    )
    logger.info(f"Starting editagent on {len(ds_selected)} Docker images.")

    # Generate a unique experiment name if not provided
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure traj_dir exists
    traj_dir_path = Path(traj_dir)
    traj_dir_path.mkdir(parents=True, exist_ok=True)

    # Generate a filename for the JSONL file
    jsonl_file = traj_dir_path / f"{exp_name}.jsonl"

    if use_existing:
        if jsonl_file.exists():
            with open(jsonl_file) as f:
                existing_dockers = []
                for line in f.readlines():
                    try:
                        existing_dockers.append(
                            Trajectory.load_from_model_dump_json(line).ds[
                                "docker_image"
                            ]
                        )
                    except:
                        print("error in jsonl file")

            ds_selected = [
                ds_entry
                for ds_entry in ds_selected
                if ds_entry["docker_image"] not in existing_dockers
            ]

    if skip_existing:
        old_jsonl_files_glob = f"{exp_name[:-1]}*"
        for old_jsonl_file in traj_dir_path.glob(old_jsonl_files_glob):
            with open(old_jsonl_file) as f:
                existing_dockers = [
                    loadline["ds"]["docker_image"]
                    for line in f
                    for loadline in [json.loads(line)]
                    if loadline["reward"] == 1
                ]

            ds_selected = [
                ds_entry
                for ds_entry in ds_selected
                if ds_entry["docker_image"] not in existing_dockers
            ]

    logger.info(
        f"Starting editagent on {len(ds_selected)} Docker images after filtering."
    )

    # Prepull all Docker images in parallel before starting main execution
    if ds_selected and prepull_images:
        logger.info("Prepulling Docker images before starting main execution...")
        prepull_docker_images(ds_selected, max_workers=max_workers)
        logger.info("Docker image prepull completed.")

    # with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor using keyword arguments
        future_to_image = {
            executor.submit(
                runagent,
                ds=ds_entry,
                exp_name=exp_name,
                max_steps=max_steps,
                num_restarts=num_restarts,
                max_steps_absolute=max_steps_absolute,
                llm_name=llm_name,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                use_fn_calling=use_fn_calling,
                backend=backend,
                max_reward_calc_time=max_reward_calc_time,
                max_iterations=max_iterations,
                scaffold=scaffold,
                max_tokens=max_tokens,
                max_completion_tokens=max_completion_tokens,
                max_total_time=max_total_time,
                max_llm_time=max_llm_time,
                max_llm_retries=max_llm_retries,
                oracle=oracle,
            ): ds_entry[
                "docker_image"
            ]  # <-- store the docker_image from ds_entry here
            for ds_entry in ds_selected
        }

        health_check_failed = False
        for future in concurrent.futures.as_completed(future_to_image):
            docker_image = future_to_image[
                future
            ]  # <-- retrieve that stored docker_image
            try:
                result = future.result()
                if result is not None:
                    with open(jsonl_file, "a") as f:
                        f.write(result + "\n")
            except HealthCheckFailureException as e:
                # Health check failed - stop all processing
                logger.critical(f"Health check failure detected for Docker image {docker_image}: {e}")
                logger.critical("Stopping all remaining tasks due to LLM server health check failure.")
                health_check_failed = True
                # Cancel all pending futures
                for pending_future in future_to_image.keys():
                    if not pending_future.done():
                        pending_future.cancel()
                        logger.info(f"Cancelled pending task for {future_to_image[pending_future]}")
                # Shutdown executor immediately
                executor.shutdown(wait=False, cancel_futures=True)
                break
            except Exception as e:
                # Use docker_image from above when logging
                logger.error(f"Exception for Docker image {docker_image}: {e}")

        if health_check_failed:
            logger.critical("Exiting runagent_multiple due to health check failure.")
            return

    logger.info(f"editagent completed on {len(ds_selected)} Docker images.")
    signal_file = Path(f"{jsonl_file}.ok")
    with open(signal_file, "w") as fout:
        fout.write("Done!")


if __name__ == "__main__":
    # Expose functions via Fire
    Fire(
        {
            "runagent": runagent,
            "runagent_multiple": runagent_multiple,
        }
    )
