import os
import json
import torch
import gc
import traceback
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from safetensors.torch import load_file
from datasets import Dataset
import fire


def load_diverse_dataset_subset(subset_size: Optional[int] = None):
    with open("results/diverse_dataset_1k.json") as f:
        dataset = json.load(f)
    if subset_size is None:
        return Dataset.from_list(dataset)
    return Dataset.from_list(dataset[:subset_size])


def calculate_fisher_information_single(
    task_name, model_name="Qwen/Qwen2.5-7B-Instruct", batch_size=1
):
    print(f"Calculating Fisher information for {task_name}")

    torch.cuda.empty_cache()
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    lora_model = get_peft_model(model, lora_config)

    adapter_path = f"models/{task_name}/adapter/adapter_model.safetensors"
    adapter_weights = load_file(adapter_path)

    lora_state_dict = {}
    for name, param in lora_model.named_parameters():
        if "lora" in name and param.requires_grad:
            adapter_name = name.replace(".default", "")
            if adapter_name in adapter_weights:
                param.data = adapter_weights[adapter_name].to(param.device, param.dtype)
                lora_state_dict[name] = param

    print(f"Found {len(lora_state_dict)} LoRA parameters to track")
    print(f"Expected: {len(adapter_weights)} adapter parameters")

    if len(lora_state_dict) == 0:
        print("ERROR: No LoRA parameters matched!")
        return {
            "status": "failed",
            "error": "No LoRA parameters matched adapter weights",
        }

    dataset = load_diverse_dataset_subset()

    fisher_info = {}
    for name, param in lora_state_dict.items():
        fisher_info[name] = torch.zeros_like(param.data)

    lora_model.train()
    total_samples = 0

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]

        texts = []
        for messages in batch["messages"]:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

        tokenized = tokenizer(
            texts, truncation=True, padding=True, max_length=1024, return_tensors="pt"
        )

        input_ids = tokenized["input_ids"].to(lora_model.device)
        attention_mask = tokenized["attention_mask"].to(lora_model.device)

        lora_model.zero_grad()

        outputs = lora_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
        )

        loss = outputs.loss
        loss.backward()

        for name, param in lora_state_dict.items():
            if param.grad is not None:
                fisher_info[name] += param.grad.data**2

        total_samples += len(batch["messages"])

        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {i + batch_size}/{len(dataset)} examples")
            torch.cuda.empty_cache()

    if total_samples > 0:
        for name in fisher_info:
            fisher_info[name] /= total_samples

    total_fisher_sum = 0
    param_count = 0
    for name, values in fisher_info.items():
        total_fisher_sum += torch.sum(values).item()
        param_count += values.numel()

    print(
        f"Debug: total_samples={total_samples}, param_count={param_count}, total_fisher_sum={total_fisher_sum}"
    )

    avg_fisher_value = total_fisher_sum / param_count if param_count > 0 else 0.0

    result_summary = {
        "status": "success",
        "total_samples": total_samples,
        "total_fisher_sum": total_fisher_sum,
        "param_count": param_count,
        "avg_fisher_value": avg_fisher_value,
    }

    os.makedirs(f"results/fisher_information/{task_name}", exist_ok=True)
    torch.save(fisher_info, f"results/fisher_information/{task_name}/fisher_info.pt")

    with open(f"results/fisher_information/{task_name}/summary.json", "w") as f:
        json.dump(result_summary, f, indent=2)

    del model, lora_model, fisher_info, adapter_weights
    torch.cuda.empty_cache()
    gc.collect()

    return result_summary


def calculate_all_fisher_information():
    known_tasks = [
        "latin_translation",
        "codesearchnet_python",
        "python_instructions_alpaca",
        "alpaca_instructions",
        "ms_marco_qa",
        "xsum",
        "reason_math",
        "gsm8k_math",
    ]

    results_summary = {}

    for task_name in known_tasks:
        print(f"\n{'='*60}")
        print(f"Processing {task_name}")
        print(f"{'='*60}")

        try:
            result = calculate_fisher_information_single(task_name)
            results_summary[task_name] = result

            if result["status"] == "success":
                print(
                    f"Completed {task_name}: avg_fisher={result['avg_fisher_value']:.6e}"
                )
            else:
                print(f"Failed {task_name}: {result['error']}")

        except Exception as e:
            print(f"Failed to process {task_name}: {e}")
            traceback.print_exc()
            results_summary[task_name] = {"status": "failed", "error": str(e)}
            torch.cuda.empty_cache()
            gc.collect()

    os.makedirs("results/fisher_information", exist_ok=True)

    with open("results/fisher_information/all_summaries.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    successful = [t for t, r in results_summary.items() if r["status"] == "success"]
    failed = [t for t, r in results_summary.items() if r["status"] == "failed"]

    print(f"\n{'='*60}")
    print("FISHER INFORMATION CALCULATION COMPLETE")
    print(f"{'='*60}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    for task in successful:
        avg_fisher = results_summary[task]["avg_fisher_value"]
        print(f"  ✓ {task}: avg_fisher={avg_fisher:.6e}")

    for task in failed:
        print(f"  ✗ {task}: {results_summary[task]['error']}")

    return results_summary


if __name__ == "__main__":
    fire.Fire(calculate_all_fisher_information)
