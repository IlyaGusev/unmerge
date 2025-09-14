Research Proposal: Unmerging via Sparse Coding (UNMERGE)

## 1. Novelty Statement

The core novelty of this work lies in establishing a **verifiable framework for model capability attribution** that allows backtracking of merged models. 

Our proposed method, UNMERGE, reframes the problem. We treat a model's fine-tuned capabilities, encapsulated in a task vector, as a "bill of materials" that can be decomposed into a sparse, non-negative combination of known "micro-task" vectors from a pre-built dictionary.

It enables a controlled setting that can be a base for furure experiments. For instance, with causal verification.

## 2. Detailed Plan of Experiments

This plan is designed to be executable on a **single consumer-grade GPU (e.g., NVIDIA RTX 3090 with 24GB VRAM)**.

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
1. Expand LoRA adapters to the base model weights space (ΔW).
2. Aggregate weight magnitutes |ΔW| across all adapters, for instance with `max`.
3. Get top-k weights per module (q, k, v, o and different layers) judging by aggregated |ΔW|.
4. Convert it to a binary mask.
5. Apply this mask for all task vectors and target vectors.

The whole process should be easily vectorized.

Result: Compressed task and target vectors. The desired size is around 100k parameters.


### Phase 4: Decomposition

Objective: Decompose the target task vectors using our proposed algorithm.
Algorithm: For each of the 30-40 target models:
1. Extract the target task vector (the difference between merged model weights and the base model)
2. As applying decomposition to all 4B parameters is infeasible, we will operate on a **per-layer basis** and only on the **non-zero parameters of the task vector**. Since LoRA vectors are inherently sparse, this is computationally tractable.
3. Apply different decomposition methods:
- LASSO
- Dot product
- Ridge regression
- Orthogonal Matching Pursuit
- NNLS
- Anything you can think of and anything extracted from the literature
4. Calculate the following metrics for the first and the second model groups (known and mixed vectors):
- Component precision
- Component recall
- Sparsity
- Number of perfect matches (when all extracted components are correct, calculate only for known vectors)
- Reconstruction error: 1 - max(0, cos(target, reconstruction))^2
5. For the third group (models composed only from unknown task vectors not from the dictionary) calculate:
- Sparsity
- Reconstruction error: 1 - max(0, cos(target, reconstruction))^2
- Semantic alignment between extracted and real task vectors.

Use the same fixed task vector dictionary and three groups of target models.

Result: This process yields comprehensive comparsions between different decompsotion methods in different conditions.
