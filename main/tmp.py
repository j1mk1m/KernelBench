import os
import json
from configs import parse_autorule_args

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_DIR, "KernelBench")

# def read_best_k_kernels(level: int):
#     with open(os.path.join(KERNEL_BENCH_PATH, f"best_k_kernels_level{level}.json"), "r") as f:
#         best_k_kernels = json.load(f)
#     return best_k_kernels

# def main(config):
#     best_k_kernels = read_best_k_kernels(config.level)
#     total_combinations = 0
#     for problem, kernels in best_k_kernels.items():
#         n = len(kernels)
#         if n >= 2:
#             combinations = n * (n - 1) // 2
#             print(f"Number of combinations in {problem}: {combinations}")
#             total_combinations += combinations
#     print(f"Total number of all possible combinations of 2 kernels per problem: {total_combinations}")


# if __name__ == "__main__":
#     config = parse_autorule_args()
#     main(config)


import sys

if len(sys.argv) < 2:
    print("Usage: python tmp.py <json_file_path>")
    sys.exit(1)

json_file_path = sys.argv[1]

with open(json_file_path, "r") as f:
    data = json.load(f)
    count = 0
    for level, level_data in data.items():
        for prob_id, prob_data in level_data.items():
            correct_count = 0
            for sample_id, eval_result in prob_data.items():
                if eval_result.get("correctness", False):
                    correct_count += 1
            if correct_count >= 2:
                count += 1
    print(f"Number of problems with at least 2 correct samples: {count}")




    # count = 0
    # for prob_id, kernels in data.items():
    #     if len(kernels["correct"]) >= 2:
    #         count += 1
    # print(f"Number of problems with less than 2 kernels: {count}")

    # count = 0
    # for level in data.values():
    #     for problem in level.values():
    #         correct_count = 0
    #         for sample_id, eval_result in problem.items():
    #             # eval_result may be a dict or object; assume dict
    #             if eval_result.get("correctness", False):
    #                 correct_count += 1
    #         if correct_count >= 2:
    #             count += 1
    # print(f"Number of problems with at least 2 correct samples: {count}")

