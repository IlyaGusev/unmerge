import os
import json
import random
import itertools
from typing import List, Dict, Any
import yaml

import fire

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

KNOWN_TASKS = [
    "latin_translation",
    "codesearchnet_python",
    "python_instructions_alpaca",
    "alpaca_instructions",
    "ms_marco_qa",
    "xsum",
    "reason_math",
    "gsm8k_math",
]

UNKNOWN_TASKS = [
    "arjun_python_qa",
    "imdb_sentiment",
    "style_transfer",
    "squad_qa",
    "orca_math",
    "stablecode_python",
    "arxiv_summarization",
]

ALL_TASKS = KNOWN_TASKS + UNKNOWN_TASKS

MERGE_METHODS = ["linear", "ties", "dare_linear", "task_arithmetic"]


def generate_task_combinations(
    tasks: List[str], min_tasks: int = 2, max_tasks: int = 5
) -> List[List[str]]:
    combinations = []
    for size in range(min_tasks, max_tasks + 1):
        for combo in itertools.combinations(tasks, size):
            combinations.append(list(combo))
    return combinations


def create_merge_config(
    model_name: str, method: str, adapters: List[str], weights: List[float] = None
) -> Dict[str, Any]:
    if weights is None:
        weights = [1.0 / len(adapters)] * len(adapters)

    models = []
    for i, adapter in enumerate(adapters):
        adapter_path = f"models/{adapter}/adapter"
        models.append(
            {
                "model": {"model": BASE_MODEL, "adapter": adapter_path},
                "parameters": {"weight": weights[i]},
            }
        )

    config = {
        "merge_method": method,
        "base_model": BASE_MODEL,
        "models": models,
        "parameters": {},
        "dtype": "bfloat16",
    }

    if method == "ties":
        config["parameters"] = {"density": 0.5, "int8_mask": True}
    elif method == "dare_linear":
        config["parameters"] = {"density": 0.5}

    return config


def generate_all_merge_configs(num_per_group: int = 6):
    configs = []

    random.seed(42)

    known_combinations = generate_task_combinations(KNOWN_TASKS)
    unknown_combinations = generate_task_combinations(UNKNOWN_TASKS)
    mixed_combinations = generate_task_combinations(ALL_TASKS)

    random.shuffle(known_combinations)
    random.shuffle(unknown_combinations)
    random.shuffle(mixed_combinations)

    known_combinations = known_combinations[:num_per_group]
    unknown_combinations = unknown_combinations[:num_per_group]
    mixed_combinations = [
        combo
        for combo in mixed_combinations
        if any(task in KNOWN_TASKS for task in combo)
        and any(task in UNKNOWN_TASKS for task in combo)
    ][:num_per_group]

    config_id = 1

    for group_name, combinations in [
        ("known", known_combinations),
        ("unknown", unknown_combinations),
        ("mixed", mixed_combinations),
    ]:
        for combo in combinations:
            for method in MERGE_METHODS:
                model_name = f"{group_name}_{method}_{len(combo)}tasks_{config_id:03d}"

                config = create_merge_config(model_name, method, combo)

                configs.append(
                    {
                        "model_name": model_name,
                        "group": group_name,
                        "method": method,
                        "tasks": combo,
                        "num_tasks": len(combo),
                        "config": config,
                        "config_id": config_id,
                    }
                )

                config_id += 1

    return configs


def save_merge_config(config_data: Dict[str, Any], config_dir: str):
    config_path = os.path.join(config_dir, f"{config_data['model_name']}.yaml")

    with open(config_path, "w") as f:
        yaml.dump(config_data["config"], f, default_flow_style=False)

    return config_path


def merge_single_model(config_data: Dict[str, Any], config_dir: str, output_dir: str):
    config_path = save_merge_config(config_data, config_dir)
    model_output_path = os.path.join(output_dir, config_data["model_name"])

    print(f"Merging {config_data['model_name']} using {config_data['method']} method")
    print(f"Tasks: {', '.join(config_data['tasks'])}")

    cmd = f"mergekit-yaml {config_path} {model_output_path} --cuda --trust-remote-code"

    result = os.system(cmd)

    if result == 0:
        metadata = {
            "model_name": config_data["model_name"],
            "group": config_data["group"],
            "method": config_data["method"],
            "tasks": config_data["tasks"],
            "num_tasks": config_data["num_tasks"],
            "config_id": config_data["config_id"],
            "base_model": BASE_MODEL,
            "merge_status": "success",
        }

        metadata_path = os.path.join(model_output_path, "merge_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Successfully merged {config_data['model_name']}")
        return True
    else:
        print(f"Failed to merge {config_data['model_name']}")
        return False


def merge_all_models():
    os.makedirs("models/merged", exist_ok=True)
    os.makedirs("results/merge_configs", exist_ok=True)

    configs = generate_all_merge_configs()

    print(f"Generated {len(configs)} merge configurations")

    results = []
    successful_merges = 0
    failed_merges = 0

    for config_data in configs:
        try:
            success = merge_single_model(
                config_data, "results/merge_configs", "models/merged"
            )

            if success:
                successful_merges += 1
            else:
                failed_merges += 1

            results.append(
                {
                    "model_name": config_data["model_name"],
                    "group": config_data["group"],
                    "method": config_data["method"],
                    "tasks": config_data["tasks"],
                    "num_tasks": config_data["num_tasks"],
                    "status": "success" if success else "failed",
                }
            )

        except Exception as e:
            print(f"Error merging {config_data['model_name']}: {str(e)}")
            failed_merges += 1
            results.append(
                {
                    "model_name": config_data["model_name"],
                    "group": config_data["group"],
                    "method": config_data["method"],
                    "tasks": config_data["tasks"],
                    "num_tasks": config_data["num_tasks"],
                    "status": "error",
                    "error": str(e),
                }
            )

    summary = {
        "total_configs": len(configs),
        "successful_merges": successful_merges,
        "failed_merges": failed_merges,
        "results": results,
    }

    with open("results/merge_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nMerging complete:")
    print(f"Total configurations: {len(configs)}")
    print(f"Successful merges: {successful_merges}")
    print(f"Failed merges: {failed_merges}")

    return summary


if __name__ == "__main__":
    fire.Fire(merge_all_models)
