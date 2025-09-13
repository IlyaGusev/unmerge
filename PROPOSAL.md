esearch Proposal: Causal Unmerging via Sparse Coding (UNMERGE)

## 1. Novelty Statement

The core novelty of this work lies in establishing a **causally verifiable framework for model capability attribution**. While prior work has explored the parameter space of neural networks to identify features or abilities (e.g., "Dissecting Model Abilities with Parameter-Space Probes", 2305.14241), these methods primarily offer correlational insights. They can identify *where* a skill resides but do not provide a mechanism to prove this attribution through direct, surgical intervention.

Our proposed method, UNMERGE, reframes the problem. We treat a model's fine-tuned capabilities, encapsulated in a task vector, as a "bill of materials" that can be decomposed into a sparse, non-negative combination of known "micro-task" vectors from a pre-built dictionary. The crucial innovation is the **subtractive, causal verification loop**: by identifying a micro-task's contribution (e.g., 'pandas programming'), we can subtract its corresponding vector from the model's weights and verify a precise, predictable performance drop on that specific skill, while demonstrating minimal impact on unrelated abilities. This moves beyond mere observation to a formal, editable, and verifiable understanding of a model's composition, which has profound implications for model safety, customisation, and interpretability. We are not merely dissecting; we are enabling a form of "model neurosurgery".

## 2. Detailed Plan of Experiments

This plan is designed to be executable on a **single consumer-grade GPU (e.g., NVIDIA RTX 3090 with 24GB VRAM)** and within a **$100 OpenRouter credit budget**.

### Phase 1: Micro-Task Dictionary Construction

Objective: Create a dictionary of ~15 distinct "micro-task" vectors.

Base Model: Qwen/Qwen2.5-7B-Instruct. Its size is manageable for an RTX 3090.

Micro-Tasks: We will fine-tune the base model on approximately 15 distinct, narrow tasks. These will be sourced from public datasets and categorized as follows:
- Python Coding (~5 tasks): Subsets of datasets like CodeAlpaca, focusing on specific libraries (e.g., `numpy` array manipulation, `pandas` DataFrame operations, `matplotlib` plotting basics).
- Structured Text Generation (~5 tasks): Tasks requiring generation of specific formats (e.g., generating valid JSON from natural language, creating XML snippets, formatting text as Markdown tables).
- Logic & Reasoning (~5 tasks): Subsets of reasoning datasets (e.g., GSM8K, Aqua-RAT) filtered for specific reasoning patterns (e.g., single-step arithmetic, multi-hop reasoning).
- Stylistic Transfer (~5 tasks): Datasets for stylistic fine-tuning (e.g., "speak like a pirate," "write in a formal, academic tone").
- Other (~5 tasks): Datasets for other tasks.

We will use half of the task vectors as a known dictionary, and half of the vectors should not be used in the dictionary.

Fine-tuning Method:
- Technique: LoRA (Low-Rank Adaptation) will be used to generate task vectors efficiently.
- Parameters: `rank=32`, `alpha=32`, `learning_rate=2e-4`, `epochs=3`. These settings are known to produce effective adaptations on an RTX 3090 within a few hours per task.
- Output: Each fine-tuning run produces a LoRA adapter. The difference `(LoRA_weights - base_model_weights)` will be saved as a "micro-task vector" in our dictionary.


### Phase 2: Target Model Creation

Objective: Create a set of multi-skill models for decomposition.
Method: We will use different merging methods: TIES-merging/DARE/Task arithmetic (from `mergekit`) to combine from 2 to 5 randomly selected micro-task LoRAs from the dictionary created in Phase 1.
Groups:
- First group: models composed only from known vectors from the dictionary.
- Second group: models mixed from known and unknown task vectors.
- Third group: models only from unknown task vectors not from the dictionary.

Result: This process yields test models whose "ground truth" composition is known, providing a perfect testbed to validate our decomposition algorithm. The merged LoRA adapter represents the "target task vector" for analysis. With the second and the third group we can simulate close to real-world scenarios.


### Parse 3: Reducing Task Vectors Dimensionality

Objective: Make task vectors tractable for decomposition.
Algorithm:
1. Expand LoRA adapters to the weights space (ΔW)
2. Get top-k weights per module judging by |ΔW|.
Result: Compressed task and target vectors. The desired size is around 100k parameters.


### Phase 4: Decomposition

Objective: Decompose the target task vectors using our proposed algorithm.
Algorithm: For each of the 30-40 target models:
1. Extract the target task vector (the difference between merged model weights and the base model)
2. As applying decomposition to all 4B parameters is infeasible, we will operate on a **per-layer basis** and only on the **non-zero parameters of the task vector**. Since LoRA vectors are inherently sparse, this is computationally tractable.
3. Apply different decomposition methods:
- LASSO
- Group LASSO
- Dot product
- Ridge regression
- Orthogonal Matching Pursuit
- Sparse auto-encoders
- Anything you can think of and anything extracted from the literature
4. Calculate the following metrics for the first and the second model groups (known and mixed vectors):
- Component precision
- Component recall
- Sparsity
- Number of perfect matches (when all extracted components are correct)
- Normalized reconstruction error: norm(w_target - w_reconstruction) / norm(w_target)
5. For the third group (models composed only from unknown task vectors not from the dictionary) calculate:
- Sparsity
- Normalized reconstruction error: norm(w_target - w_reconstruction) / norm(w_target)
- Semantic alignment between extracted and real task vectors.

Use the same fixed task vector dictionary and three groups of target models.

Result: This process yields comprehensive comparsions between different decompsotion methods in different conditions.

### Phase 5: Causal Verification

Objective: Surgically remove a decomposed capability and measure the impact, validating the causal link.
Protocol:
1. For a target model and a decomposed skill with a high coefficient (e.g., 'pandas programming'), retrieve the corresponding micro-task vector from the dictionary.
2. Surgical Subtraction: Subtract the weighted micro-task vector (`c_i * ΔW_dict_i`) from the target model's weights.
3. This creates a new, "ablated" model.

Evaluation Metrics:
- Primary Metric (Capability Drop): We will evaluate the ablated model on a held-out test set specific to the removed skill (e.g., a set of 50 pandas programming problems not seen during training). We expect to see a significant, measurable drop in performance.
- Secondary Metric (Stability): We will evaluate the ablated model on a broad suite of unrelated benchmarks (e.g., MMLU, standard GSM8K). We expect to see minimal to no performance degradation, demonstrating the "surgical" nature of the intervention and the absence of catastrophic forgetting.

