import os
import json

import fire
import torch
from safetensors.torch import load_file
from transformers import AutoModel, AutoConfig


def load_base_model_weights():
    print("Loading base model weights...")
    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    try:
        model = AutoModel.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16)
        base_weights = {}
        for name, param in model.named_parameters():
            if any(n in name for n in ("q_proj", "v_proj", "k_proj", "o_proj")):
                base_weights[name] = param.detach().clone()
        del model
        torch.cuda.empty_cache()
        return base_weights
    except Exception as e:
        print(f"Error loading base model: {e}")
        return None


def extract_target_vector_from_merged(merged_model_path, base_weights):
    print(f"Extracting target vector from {merged_model_path}")

    merged_model = AutoModel.from_pretrained(merged_model_path, torch_dtype=torch.bfloat16)
    target_vector = {}

    for name, param in merged_model.named_parameters():
        if name in base_weights:
            delta = param.detach().clone() - base_weights[name]
            if delta.sum() != 0:
                print(merged_model_path, name, delta.sum())
            target_vector[name] = delta
        else:
            print(f"Missing: {name}")

    del merged_model
    torch.cuda.empty_cache()
    return target_vector


def apply_binary_mask(weights, binary_mask):
    compressed_weights = []
    for name, mask in sorted(binary_mask.items()):
        name = name.replace("model.", "")
        name += ".weight"
        if name in weights:
            weight = weights[name]
            compressed_weight = weight[mask]
            compressed_weights.append(compressed_weight.flatten())
    final_vector = torch.cat(compressed_weights, dim=0)
    return final_vector


def compress_target_vectors(mask_path: str = "models/unified_selection_mask.pt"):
    print("Starting target vector extraction and compression...")
    if not os.path.exists(mask_path):
        print(f"Binary mask not found at {mask_path}")
        return

    binary_mask = torch.load(mask_path)
    print(f"Loaded binary mask with {len(binary_mask)} entries")

    base_weights = load_base_model_weights()
    if base_weights is None:
        return

    print(f"Loaded base weights for {len(base_weights)} parameters")

    merge_results_path = "results/merge_results.json"
    assert os.path.exists(merge_results_path)

    with open(merge_results_path) as f:
        merge_data = json.load(f)
        merge_results = merge_data["results"]

    compressed_targets = {}
    success_count = 0

    for i, result in enumerate(merge_results):
        model_name = result["model_name"]
        merged_path = f"models/merged/{model_name}"

        if not os.path.exists(merged_path):
            print(f"Merged model not found: {merged_path}")
            continue

        print(f"Processing {i+1}/{len(merge_results)}: {model_name}")

        target_vector = extract_target_vector_from_merged(merged_path, base_weights)
        if target_vector is None:
            continue

        compressed_vector = apply_binary_mask(target_vector, binary_mask)
        output_path = f"models/merged/{model_name}/compressed_vector.pt"
        torch.save(compressed_vector, output_path)

        compressed_targets[model_name] = {
            "path": output_path,
            "selected_params": len(compressed_vector),
            "group": result["group"],
            "method": result["method"],
            "tasks": result["tasks"]
        }

        success_count += 1
        print(f"  Saved with {len(compressed_vector)} selected parameters")

    metadata = {
        "compressed_target_vectors": compressed_targets,
        "total_processed": success_count,
        "binary_mask_path": mask_path
    }

    with open("results/compress_target_vectors_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Successfully processed {success_count} target vectors")
    return metadata


if __name__ == "__main__":
    fire.Fire(compress_target_vectors)
