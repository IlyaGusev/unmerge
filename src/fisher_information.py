import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import fire


class FisherInformationComputer:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading model {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

    def prepare_chat_data(self, examples, max_length=512):
        tokenized_inputs = []

        for example in examples:
            messages = example["messages"]

            chat_text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )

            tokens = self.tokenizer(
                chat_text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors="pt"
            )

            tokenized_inputs.append({
                'input_ids': tokens['input_ids'].squeeze(0),
                'attention_mask': tokens['attention_mask'].squeeze(0)
            })

        return tokenized_inputs

    def collate_fn(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )

        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks
        }

    def compute_fisher_information(self, dataset_path, batch_size=8, max_batches=None):
        print(f"Loading dataset from {dataset_path}")

        with open(dataset_path, 'r') as f:
            examples = json.load(f)

        print(f"Loaded {len(examples)} examples")

        tokenized_data = self.prepare_chat_data(examples)
        dataloader = DataLoader(
            tokenized_data, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=self.collate_fn
        )

        fisher_info = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param.data)

        num_batches = len(dataloader)
        if max_batches:
            num_batches = min(num_batches, max_batches)

        print(f"Computing Fisher information over {num_batches} batches...")

        total_samples = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing Fisher")):
            if max_batches and batch_idx >= max_batches:
                break

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            self.model.zero_grad()

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_info[name] += param.grad.data ** 2

            total_samples += input_ids.size(0)

        for name in fisher_info:
            fisher_info[name] /= total_samples

        print(f"Fisher information computed over {total_samples} samples")
        return fisher_info

    def select_important_parameters(self, fisher_info, target_params=100000, percentile_threshold=95):
        all_fisher_values = []
        param_locations = {}

        for param_name, fisher_tensor in fisher_info.items():
            flat_fisher = fisher_tensor.flatten()
            for i, value in enumerate(flat_fisher):
                all_fisher_values.append(value.item())
                param_locations[len(all_fisher_values) - 1] = (param_name, i)

        fisher_array = np.array(all_fisher_values)

        lower_clip = np.percentile(fisher_array, 2.5)
        upper_clip = np.percentile(fisher_array, 97.5)
        fisher_clipped = np.clip(fisher_array, lower_clip, upper_clip)

        if target_params < len(fisher_clipped):
            threshold_idx = len(fisher_clipped) - target_params
            threshold = np.partition(fisher_clipped, threshold_idx)[threshold_idx]
        else:
            threshold = np.percentile(fisher_clipped, percentile_threshold)

        important_indices = np.where(fisher_clipped >= threshold)[0]

        important_params = {}
        for idx in important_indices:
            param_name, param_idx = param_locations[idx]
            if param_name not in important_params:
                important_params[param_name] = []
            important_params[param_name].append(param_idx)

        important_param_tensors = {}
        total_selected = 0

        for param_name, indices in important_params.items():
            param_shape = fisher_info[param_name].shape
            tensor_indices = []
            for flat_idx in indices:
                tensor_idx = np.unravel_index(flat_idx, param_shape)
                tensor_indices.append(tensor_idx)

            important_param_tensors[param_name] = tensor_indices
            total_selected += len(tensor_indices)

        print(f"Selected {total_selected} important parameters from {len(fisher_array)} total")
        print(f"Selected parameters from {len(important_param_tensors)} layers")

        return important_param_tensors, threshold

    def save_fisher_results(self, fisher_info, important_params, threshold, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        fisher_cpu = {}
        for name, tensor in fisher_info.items():
            fisher_cpu[name] = tensor.cpu().float()

        torch.save(fisher_cpu, os.path.join(output_dir, "fisher_information.pt"))

        with open(os.path.join(output_dir, "important_params.json"), 'w') as f:
            important_params_serializable = {}
            for param_name, indices in important_params.items():
                important_params_serializable[param_name] = [list(idx) for idx in indices]
            json.dump(important_params_serializable, f, indent=2)

        metadata = {
            "total_parameters_selected": sum(len(indices) for indices in important_params.values()),
            "num_layers_selected": len(important_params),
            "selection_threshold": float(threshold),
            "parameter_shapes": {name: list(tensor.shape) for name, tensor in fisher_info.items()},
            "layers_selected": list(important_params.keys())
        }

        with open(os.path.join(output_dir, "fisher_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Fisher information results saved to {output_dir}")
        print(f"Total parameters selected: {metadata['total_parameters_selected']}")
        print(f"Layers involved: {metadata['num_layers_selected']}")


def compute_fisher_information_main(
    dataset_path="results/diverse_dataset_5k.json",
    output_dir="results/fisher_information",
    batch_size=4,
    max_batches=100,
    target_params=100000
):
    print("Starting Fisher Information computation...")

    computer = FisherInformationComputer()

    fisher_info = computer.compute_fisher_information(
        dataset_path=dataset_path,
        batch_size=batch_size,
        max_batches=max_batches
    )

    important_params, threshold = computer.select_important_parameters(
        fisher_info, target_params=target_params
    )

    computer.save_fisher_results(fisher_info, important_params, threshold, output_dir)

    return fisher_info, important_params


if __name__ == "__main__":
    fire.Fire(compute_fisher_information_main)
