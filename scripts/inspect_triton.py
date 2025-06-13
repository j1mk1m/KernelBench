import torch
from torch.profiler import profile, record_function, ProfilerActivity
import logging
import os
import io


"""
[WIP] For debugging and analysis
Inspect torch compile generated triton code
as well as generate flamegraph for a particular problem when executed with Torch Eager/Compile
using PyTorch Profiler
"""

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

device = "cuda:0"


from src.utils import read_file
from src.eval import (
    load_custom_model,
    load_original_model_and_inputs,
    time_execution_with_cuda_event,
    get_timing_stats,
    set_seed,
)

def fetch_ref_arch_from_dataset(dataset: list[str], 
                                problem_id: int) -> tuple[str, str, str]:
    """
    Fetch the reference architecture from the problem directory
    problem_id should be logical index (1-indexed), matching the problem_id in the problem_name

    Returns:
        ref_arch_path: str, the path to the reference architecture
        ref_arch_name: str, the name of the reference architecture
        ref_arch_src: str, the source code of the reference architecture
    """
    ref_arch_path = None
    
    for file in dataset:
        if file.split("/")[-1].split("_")[0] == str(problem_id):
            ref_arch_path = file
            break
    if ref_arch_path is None:
        raise ValueError(f"No reference architecture found for problem_id {problem_id}")
    
    ref_arch_src = read_file(ref_arch_path)

    ref_arch_name = ref_arch_path.split("/")[-1]
    return (ref_arch_path, ref_arch_name, ref_arch_src)


def run_profile_and_save_trace(dataset: list[str], problem_id: int, num_trials=10):
    """
    Helper function to get Torch Profile of a problem
    # TODO: Fix up this function
    """
    ref_arch_path, ref_arch_name, ref_arch_src = fetch_ref_arch_from_dataset(
        dataset, problem_id
    )
    context = {}
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        ref_arch_src, context
    )
    # try:
    with torch.no_grad():
        profiling_scheduler = torch.profiler.schedule(
            wait=1,
            warmup=2,
            active=7,
        )
        torch.cuda.synchronize(device=device)
        set_seed(42)
        inputs = get_inputs()
        set_seed(42)
        init_inputs = get_init_inputs()
        inputs = [
            x.cuda(device=device) if isinstance(x, torch.Tensor) else x
            for x in inputs
        ]
        init_inputs = [
            x.cuda(device=device) if isinstance(x, torch.Tensor) else x
            for x in init_inputs
        ]
        
        # Create base model
        model = Model(*init_inputs)
        model = model.cuda(device=device)
        
        # Profile non-compiled model
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=profiling_scheduler
        ) as prof:
            with record_function("non_compiled_forward"):
                for _ in range(num_trials):
                    model(*inputs)
                    prof.step()
        print(f"\nProfiling results for non-compiled model:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        # Profile compiled model
        model_compiled = torch.compile(model)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        ) as prof_compiled:
            with record_function("compiled_forward"):
                for _ in range(num_trials):
                    model_compiled(*inputs)
                    prof_compiled.step()
        print(f"\nProfiling results for compiled model:")
        print(prof_compiled.key_averages().table(sort_by="cuda_time_total", row_limit=10))


        prof.export_chrome_trace("trace_non_compiled.json")
        prof_compiled.export_chrome_trace("trace_compiled.json")

    # except Exception as e:
        # print(f"[Eval] Error in Measuring Performance: {e}")

def get_torch_compile_triton(level_num, problem_id):
    """
    Get the triton code generated by torch compile for a particular problem
    """
    ref_arch_path, ref_arch_name, ref_arch_src = fetch_ref_arch_from_dataset(
        dataset, problem_id, with_name=True
    )
    context = {}
    # import pdb; pdb.set_trace()
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        ref_arch_src, context
    )
    try:
        with torch.no_grad():
            torch.cuda.synchronize(device=device)
            set_seed(42)
            inputs = get_inputs()
            set_seed(42)
            init_inputs = get_init_inputs()
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]
            init_inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in init_inputs
            ]
            model = Model(*init_inputs)

            # output triton code
            log_file = f"results/triton_code/level{level_num}_problem_{problem_id}_triton.log"
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            # TODO: Figure out a way to save to a file 

            torch._logging.set_logs(output_code=True)

            # Call torch compile
            model =torch.compile(model, backend="inductor")

            # reduce overhead -> 
            # model = torch.compile(model, mode="")
            
            model = model.cuda(device=device)
            

            torch.cuda.synchronize(device=device)
            elapsed_times = time_execution_with_cuda_event(
                model, *inputs, num_trials=1, verbose=False, device=device
            )
            runtime_stats = get_timing_stats(elapsed_times, device=device)
            # json_results[f"level{level_num}"][ref_arch_name] = runtime_stats
            print(f"{ref_arch_name} {runtime_stats}")
            return (ref_arch_name)
    except Exception as e:
        print(f"[Eval] Error in Measuring Performance: {e}")
