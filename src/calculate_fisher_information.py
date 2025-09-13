import json
import torch
import gc
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import fire


def load_diverse_dataset_subset(subset_size: Optional[int] = None):
    with open("results/diverse_dataset_1k.json") as f:
        dataset = json.load(f)
    if subset_size is None:
        return Dataset.from_list(dataset)
    return Dataset.from_list(dataset[:subset_size])


def calculate_fisher_information_single(
    model_name="Qwen/Qwen2.5-7B-Instruct", batch_size=1, active="B"
):
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
    lora_A, lora_B = {}, {}
    for n, p in lora_model.named_parameters():
        if "lora_A" in n:
            lora_A[n] = p
            p.requires_grad_(True)
        elif "lora_B" in n:
            lora_B[n] = p
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    with torch.no_grad():
        if active == "B":
            for p in lora_B.values():
                p.zero_()
            for n, p in lora_A.items():
                IN = p.shape[1]
                p.normal_(0, 1.0 / IN**0.5)
        else:
            for p in lora_A.values():
                p.zero_()
            for n, p in lora_B.items():
                OUT = p.shape[0]
                p.normal_(0, 1.0 / OUT**0.5)

    lora_model.eval()
    print(f"Found {len(lora_A) + len(lora_B)} LoRA parameters to track")

    dataset = load_diverse_dataset_subset()
    fisher_info = {}
    if active == "A":
        for name, param in lora_A.items():
            fisher_info[name] = torch.zeros_like(param.data)
    elif active == "B":
        for name, param in lora_B.items():
            fisher_info[name] = torch.zeros_like(param.data)

    total_tokens = 0

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

        labels = input_ids.masked_fill(attention_mask == 0, -100)
        outputs = lora_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        loss = outputs.loss
        num_tokens = (labels != -100).sum().item()
        loss.backward()

        for name, param in (lora_A if active == "A" else lora_B).items():
            if param.grad is not None:
                fisher_info[name] += (param.grad.data**2) * num_tokens

        total_tokens += num_tokens

        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {i + batch_size}/{len(dataset)} examples")
            torch.cuda.empty_cache()

    if total_tokens > 0:
        for name in fisher_info:
            fisher_info[name] /= max(total_tokens, 1)

    total_fisher_sum = 0
    param_count = 0
    for name, values in fisher_info.items():
        total_fisher_sum += torch.sum(values).item()
        param_count += values.numel()

    print(
        f"Debug: total_tokens={total_tokens}, param_count={param_count}, total_fisher_sum={total_fisher_sum}"
    )
    avg_fisher_value = total_fisher_sum / param_count if param_count > 0 else 0.0
    result_summary = {
        "status": "success",
        "total_samples": total_tokens,
        f"total_fisher_sum_{active}": total_fisher_sum,
        f"param_count_{active}": param_count,
        f"avg_fisher_value_{active}": avg_fisher_value,
    }
    return fisher_info, result_summary


def calculate_all_fisher_information():
    info_A, summary_A = calculate_fisher_information_single(active="A")
    info_B, summary_B = calculate_fisher_information_single(active="B")
    assert len(set(info_A).intersection(set(info_B))) == 0
    info = {**info_A, **info_B}
    summary = {**summary_A, **summary_B}
    torch.save(info, "results/fisher_info.pt")
    with open("results/fisher_info_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    fire.Fire(calculate_all_fisher_information)
