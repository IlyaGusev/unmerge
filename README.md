# Current pipeline

```
# Phase 1
python3 -m src.process_datasets
python3 -m src.train_adapters
python3 -m src.validate_adapters

# Phase 2
python3 -m src.merge_models

# Phase 3
python3 -m src.compress_task_vectors
python3 -m src.compress_target_vectors

# Phase 4
python3 -m src.create_ground_truth
python3 -m src.run_decomposition_experiments
```
