from dataclasses import dataclass
import random
import time

from tqdm import tqdm

import shutil
from src.eval import build_compile_cache
from src import utils as utils
import torch
import os
import multiprocessing as mp

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUNS_DIR = os.path.join(REPO_ROOT, "runs")
KERNEL_EVAL_BUILD_DIR = "/data/user_data/gyeongwk/Kernel-Coder/cache"

"""
Compile and Cache

This module contains the logic for compiling and caching the kernels
on CPU in parallel so you can speedup the evaluation process

The cache build directory must match the ones you use during evaluation phase
""" 

@dataclass
class WorkArgs:
    evaluation_id: str  # same as solution_id
    device: torch.device

def compile_single_sample(work_args: WorkArgs, config: dict, run_dir=None) -> tuple[bool, str]:
    evaluation_id = work_args.evaluation_id
    verbose = config["verbose"]
    
    utils.set_gpu_arch(config["gpu_arch"])

    build_dir = os.path.join(KERNEL_EVAL_BUILD_DIR, config["run_name"], evaluation_id)

    if run_dir is None:
        run_dir = os.path.join(RUNS_DIR, config["run_name"])
    kernel_src_path = os.path.join(run_dir, f"{evaluation_id}_kernel.py")

    if not os.path.exists(kernel_src_path):
        print(f"[ERROR] Kernel source file not found for Evaluation ID: {evaluation_id}")
        return False, "Kernel source file not found", "Kernel source file not found"

    with open(kernel_src_path, "r") as f:
        kernel_src = f.read()

    try:
        compiled_and_cached, stdout_content, error_msg = build_compile_cache(custom_model_src=kernel_src,
                                                       verbose=verbose, 
                                                       build_dir=build_dir)

        return compiled_and_cached, stdout_content, error_msg
    except Exception as e:
        print(f"[WARNING] Last level catch on {evaluation_id}: Some issue while compiling and attempting to cache for kernel: {e} ")
        return None, str(e), str(e)
    
def remove_cache_dir(config, evaluation_id: str):
    """
    Remove the cached folder for sample compilation so it can start a clean build next time
    useful for time out, failed build, etc.
    """
    cache_dir = os.path.join(KERNEL_EVAL_BUILD_DIR, config["run_name"], evaluation_id)
    print(f"cache_dir to remove: {cache_dir}")
    if os.path.exists(cache_dir):
        try:
            # Add error handling and retry with force
            shutil.rmtree(cache_dir, ignore_errors=True)
            print(f"\n[INFO] Removed cached folder for Evaluation ID: {evaluation_id}")
        except Exception as e:
            print(f"\n[WARNING] Failed to remove cache directory {cache_dir}: {str(e)}")

def batch_compile(total_work: list[str], config: dict, run_dir=None):
    """
    Batch compile cache across CPUs, assume config has num_cpu_workers
    """
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    assert "num_cpu_workers" in config, "num_cpu_workers must be specified in config for batch compile"
    try:
        with mp.Pool(config["num_cpu_workers"]) as pool:
            # Create work args for each task
            # total_work is a list of evaluation_ids (solution_id strings)
            work_args = [
                (WorkArgs(evaluation_id=evaluation_id, device=None), config, run_dir)
                for evaluation_id in total_work
            ]


            # Launch all tasks in parallel and track start times
            async_results = []
            start_times = {}
            for i, work_arg in enumerate(work_args):
                # this is type AsyncResult
                async_result = pool.apply_async(compile_single_sample, args=work_arg)
                async_results.append(async_result)
                start_times[id(async_result)] = time.time()
            
            results = []
            pending_tasks = list(enumerate(async_results))
            
            with tqdm(total=len(work_args), desc="Compile & Cache Progress") as pbar:
                while pending_tasks:
                    remaining_tasks = []
                    for i, async_result in pending_tasks:
                        try:
                            evaluation_id = total_work[i]
                            if async_result.ready():
                                try:
                                    compiled, stdout_content, error_msg = async_result.get(timeout=1)  # Short timeout for completed tasks
                                    
                                    print(f"[Status] Compilation {compiled} for Evaluation ID: {evaluation_id}")
                                    results.append((i, compiled))

                                    if not compiled:
                                        # Remove the cached folder for this failed compile so it can start a clean build next time
                                        remove_cache_dir(config, evaluation_id)
                                        
                                    pbar.update(1)
                                except Exception as e:
                                    with open("error_log.txt", "a") as f:
                                        f.write(f"\n[ERROR] Task failed for Evaluation ID: {evaluation_id}: {str(e)}")
                                    print(f"\n[ERROR] Task failed for Evaluation ID: {evaluation_id}: {str(e)}")
                                    remove_cache_dir(config, evaluation_id)
                                    results.append((i, None))
                                    pbar.update(1)
                            else:
                                # Check if the task has exceeded timeout
                                if time.time() - start_times[id(async_result)] > config["timeout"]:
                                    print(f"\n[TIME OUT] Task timed out for Evaluation ID: {evaluation_id}")
                                    
                                    remove_cache_dir(config, evaluation_id)

                                    # if we were to retry!
                                    # Start a new task for the same work
                                    print(f"Retrying for Evaluation ID: {evaluation_id}")
                                    new_async_result = pool.apply_async(compile_single_sample, args=work_args[i])
                                    start_times[id(new_async_result)] = time.time()
                                    remaining_tasks.append((i, new_async_result))
                                else:
                                    # keep going 
                                    remaining_tasks.append((i, async_result))

                        except Exception as e:
                            print(f"\n[ERROR] Unexpected error for Evaluation ID: {evaluation_id}: {str(e)}")
                            
                            remove_cache_dir(config, evaluation_id)
                            
                            results.append((i, None))
                            
                            pbar.update(1)
                    
                    pending_tasks = remaining_tasks
                    time.sleep(0.1)  # Prevent busy waiting
            
            # Sort results back to original order
            sorted_results = [r for _, r in sorted(results, key=lambda x: x[0])]
            return sorted_results

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Terminating workers...")
        pool.terminate()
        raise
    finally:
        if 'pool' in locals():
            pool.close()