import os
import json

import torch
from safetensors.torch import load_file
import fire


def extract_task_vector_corrected(adapter_path, output_path, task_name):
    print(f"Extracting task vector for {task_name}")

    adapter_weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
    print(f"Loading adapter from: {adapter_path}")

    lora_weights = load_file(adapter_weights_path)

    task_vector = {}
    total_params = 0
    vector_norm_squared = 0
    for name, weight in lora_weights.items():
        task_vector[name] = weight.clone()
        total_params += weight.numel()
        vector_norm_squared += torch.sum(weight**2).item()

    vector_norm = torch.sqrt(torch.tensor(vector_norm_squared)).item()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(task_vector, output_path)
    metadata = {
        "task_name": task_name,
        "adapter_path": adapter_path,
        "total_parameters": total_params,
        "vector_norm": vector_norm,
        "parameter_names": list(lora_weights.keys()),
    }

    metadata_path = output_path.replace(".pt", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Task vector extracted for {task_name}")
    print(f"Total parameters: {total_params}")
    print(f"Vector norm: {vector_norm:.6f}")
    print(f"Parameter types: {len(lora_weights)} tensors")
    print(f"Saved to: {output_path}")

    return metadata


def extract_task_vectors(training_results_path: str):
    with open(training_results_path) as r:
        training_results = json.load(r)

    results = []
    for run in training_results:
        task_name = run["task_name"]
        adapter_path = run["adapter_path"]
        assert adapter_path
        print(f"\n{'='*60}")
        print(f"Processing {task_name}")
        print(f"{'='*60}")

        try:
            output_path = f"models/{task_name}/task_vector.pt"
            metadata = extract_task_vector_corrected(
                adapter_path, output_path, task_name
            )
            metadata["status"] = "success"
            results.append(metadata)

        except Exception as e:
            print(f"Failed to extract task vector for {task_name}: {e}")
            results.append(
                {"task_name": task_name, "status": "failed", "error": str(e)}
            )

    summary_path = "results/extract_task_vectors_results.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    print(f"\n{'='*60}")
    print("CORRECTED TASK VECTOR EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    for r in successful:
        print(
            f"✓ {r['task_name']}: Norm={r['vector_norm']:.6f}, Params={r['total_parameters']}"
        )

    for r in failed:
        print(f"✗ {r['task_name']}: {r['error']}")

    return results


if __name__ == "__main__":
    fire.Fire(extract_task_vectors)
