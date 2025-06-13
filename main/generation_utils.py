import os

from src.utils import extract_first_code, read_file, maybe_multithread

from configs import TestTimeScalingConfig
from utils import WorkArgs, fetch_ref_arch_from_problem_id, check_if_kernel_exists
from prompts import generate_prompt


def generate_sample_single(work: WorkArgs, config: TestTimeScalingConfig, dataset, inference_server: callable, run_dir: str) -> bool:
    ref_arch_src = fetch_ref_arch_from_problem_id(dataset, work.problem_id, config.dataset_src)

    # Construct Prompt   
    custom_cuda_prompt = generate_prompt(work, config, ref_arch_src, inference_server, run_dir)
    if config.log_prompt:
        prompt_path = os.path.join(run_dir, f"level_{config.level}_problem_{work.problem_id}_sample_{work.sample_id}_prompt.txt")
        with open(prompt_path, "w") as f:
            f.write(custom_cuda_prompt)

    # Query server with constructed prompt
    custom_cuda = inference_server(custom_cuda_prompt)
    custom_cuda = extract_first_code(custom_cuda, ["python", "cpp"])
    # check LLM is able to generate custom CUDA code
    assert custom_cuda is not None, "Custom CUDA code generation failed"

    if config.verbose:
        print(f"Generated sample {work.sample_id} for problem {work.problem_id}")

    # Store to local file
    kernel_path = os.path.join(run_dir, f"level_{config.level}_problem_{work.problem_id}_sample_{work.sample_id}_kernel.py")
    with open(kernel_path, "w") as f:
        f.write(custom_cuda)
    
    return True

def generate_sample_launcher(work: WorkArgs, config: TestTimeScalingConfig, dataset, inference_server: callable, run_dir: str):
    try:
        return generate_sample_single(work, config, dataset, inference_server, run_dir)
    except Exception as e:
        print(f"Error generating problem {work.problem_id} sample {work.sample_id}: {e}")
        
        return None


def batch_generate(
    total_work: list[WorkArgs],
    config: TestTimeScalingConfig,
    dataset,
    inference_server: callable,
    run_dir: str,
):
    total_work = [work for work in total_work if not check_if_kernel_exists(run_dir, config.level, work.problem_id, work.sample_id)]
    return maybe_multithread(generate_sample_launcher, 
                      total_work, 
                      config.num_workers, 
                      pbar_name=f"Generation {config.method} Progress",
                      time_interval=config.api_query_interval, 
                      # extra args
                      config=config, 
                      dataset=dataset, 
                      inference_server=inference_server,
                      run_dir=run_dir
                      )

