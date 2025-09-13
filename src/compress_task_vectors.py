import os
import json
from collections import defaultdict

import fire
import torch
from safetensors.torch import load_file
from transformers import AutoConfig


def expand_lora_to_full_weights(lora_weights, base_model_config):
    expanded_weights = {}

    for name, weight in lora_weights.items():
        if "lora_A" in name:
            base_name = name.replace(".lora_A.weight", "")
            lora_b_name = name.replace("lora_A", "lora_B")

            if lora_b_name in lora_weights:
                lora_a = weight
                lora_b = lora_weights[lora_b_name]

                delta_w = lora_b @ lora_a

                clean_name = base_name.replace("base_model.model.", "")
                expanded_weights[clean_name] = delta_w

    return expanded_weights


def aggregate_weight_magnitudes(expanded_weights_list):
    aggregated = {}

    for expanded_weights in expanded_weights_list:
        for name, weight in expanded_weights.items():
            weight_mag = torch.abs(weight)
            if name not in aggregated:
                aggregated[name] = weight_mag
            else:
                aggregated[name] = torch.maximum(aggregated[name], weight_mag)

    return aggregated


def create_binary_mask(aggregated_weights, target_params=100000):
    module_weights = defaultdict(list)

    for name, weight in aggregated_weights.items():
        if any(proj in name for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            layer_match = name.split(".")
            layer_num = None
            for i, part in enumerate(layer_match):
                if part == "layers" and i + 1 < len(layer_match):
                    layer_num = layer_match[i + 1]
                    break

            if layer_num is not None:
                proj_type = None
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    if proj in name:
                        proj_type = proj
                        break

                if proj_type:
                    module_key = f"layer_{layer_num}_{proj_type}"
                    flat_weight = weight.flatten()
                    indices = torch.arange(len(flat_weight))
                    module_weights[module_key].append(
                        (name, flat_weight, indices, weight.shape)
                    )

    total_modules = len(module_weights)
    assert total_modules != 0, "No valid modules found"

    k_per_module = target_params // total_modules
    print(
        f"Target {target_params} params across {total_modules} modules = ~{k_per_module} params per module"
    )

    binary_mask = {}
    total_selected = 0

    for module_key, weight_list in module_weights.items():
        if not weight_list:
            continue

        combined_weights = []
        combined_info = []

        for name, flat_weight, indices, orig_shape in weight_list:
            combined_weights.append(flat_weight)
            combined_info.extend(
                [(name, i, orig_shape) for i in range(len(flat_weight))]
            )

        if combined_weights:
            all_weights = torch.cat(combined_weights)
            _, top_indices = torch.topk(
                all_weights, min(k_per_module, len(all_weights))
            )

            module_mask = {}
            for idx in top_indices:
                name, orig_idx, orig_shape = combined_info[idx]
                if name not in module_mask:
                    module_mask[name] = torch.zeros(orig_shape, dtype=torch.bool)

                flat_mask = module_mask[name].flatten()
                flat_mask[orig_idx] = True
                module_mask[name] = flat_mask.reshape(orig_shape)

            binary_mask.update(module_mask)
            total_selected += len(top_indices)

    print(f"Created binary mask with {total_selected} selected parameters")
    return binary_mask


def apply_binary_mask(weights, binary_mask):
    compressed_weights = []
    for name, weight in sorted(weights.items()):
        if name in binary_mask:
            mask = binary_mask[name]
            compressed_weight = weight[mask]
            compressed_weights.append(compressed_weight.flatten())
    final_vector = torch.cat(compressed_weights, dim=0)
    return final_vector


def compress_task_vectors(
    target_params: int = 100000,
    binary_mask_path: str = "models/unified_selection_mask.pt",
):
    required_tasks = [
        "latin_translation",
        "codesearchnet_python",
        "python_instructions_alpaca",
        "alpaca_instructions",
        "ms_marco_qa",
        "xsum",
        "reason_math",
        "gsm8k_math",
    ]

    print(f"Loading {len(required_tasks)} required task adapters...")

    base_config = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    expanded_weights_list = []
    task_vectors = {}

    for task in required_tasks:
        adapter_path = f"models/{task}/adapter/adapter_model.safetensors"
        print(f"Processing {task}...")

        assert os.path.exists(adapter_path)
        lora_weights = load_file(adapter_path)
        expanded = expand_lora_to_full_weights(lora_weights, base_config)
        expanded_weights_list.append(expanded)
        task_vectors[task] = expanded

        total_params = sum(w.numel() for w in expanded.values())
        print(f"  Expanded {task}: {total_params} parameters")

    print(f"Loaded {len(expanded_weights_list)} task vectors")

    print("Aggregating weight magnitudes across all adapters...")
    aggregated = aggregate_weight_magnitudes(expanded_weights_list)

    print("Creating binary selection mask...")
    binary_mask = create_binary_mask(aggregated, target_params=target_params)

    print("Saving binary mask...")
    torch.save(binary_mask, binary_mask_path)

    print("Applying mask to compress task vectors...")
    compressed_task_vectors = {}

    for task, weights in task_vectors.items():
        compressed = apply_binary_mask(weights, binary_mask)
        compressed_task_vectors[task] = compressed
        print(f"  {task}: {len(compressed)} parameters")

        torch.save(compressed, f"models/{task}/compressed_task_vector.pt")

    metadata = {
        "required_tasks": required_tasks,
        "tasks_processed": list(task_vectors.keys()),
        "target_compressed_params": target_params,
        "binary_mask_path": binary_mask_path,
        "compressed_task_vectors": {
            task: f"models/{task}/compressed_task_vector.pt"
            for task in task_vectors.keys()
        },
        "total_mask_entries": len(binary_mask),
        "mask_selection_method": "top-k per module (q_proj, k_proj, v_proj, o_proj)",
        "aggregation_method": "max magnitude across adapters",
    }

    with open("results/compress_task_vectors_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Compression completed successfully!")
    return metadata


if __name__ == "__main__":
    fire.Fire(compress_task_vectors)
