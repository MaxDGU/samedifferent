# Project Goals and Experiment Summary

This document outlines the goals, structure, and status of the experiments within the `metasamedifferent` repository.

## Overall Goal

Investigate the effectiveness of meta-learning compared to standard supervised learning (SGD) for same-different visual classification tasks, using various datasets and model architectures.

## Shared Resources

*   **Data:**
    *   Puebla-Bowers (PB) Shapes: Located within `data/meta_h5/` and `data/vanilla_h5/` (Specific PB dataset files need confirmation). Used for shape-based same-different tasks.
    *   Naturalistic Images: Located within `data/meta_h5/` and `data/vanilla_h5/` (Specific naturalistic dataset files need confirmation). Used for real-world object same-different tasks.
*   **Models:**
    *   Architectures: `conv2`, `conv4`, `conv6` (Specific implementation details in respective scripts/folders).
    *   Note: Meta-learning experiments often use scripts named `conv2lr.py`, `conv4lr.py`, etc., which import the base `convX` architectures but apply meta-learning training procedures.
    *   Trained Weights: Checkpoints are likely stored within experiment-specific `results/` directories or potentially a shared `models/` directory (e.g., `models/model_checkpoint_epoch_40.pt`). Exact locations vary per experiment run.

## Repository Structure Notes

*   **Local vs. Remote:** The project exists locally and on the Della cluster. The `remote/` directory contains code intended for/copied from Della.
*   **Current Focus:** Organizing and debugging the `remote/` directory to ensure it's modular and functional before transferring back to Della for GPU-accelerated runs.

## Experiments

### 1. Baselines (`baselines/`)

*   **Goal:** Establish baseline performance using standard supervised learning (SGD).
*   **Models:** `conv2`, `conv4`, `conv6`.
*   **Data:** PB Shapes (`data/vanilla_h5/`).
*   **Training:** Regular SGD.
*   **Task:** Same-Different Classification.
*   **Status:** Completed.===== Processing Architecture: conv2 ====

### 2. Meta Baseline (`meta_baseline/`)

*   **Goal:** Compare meta-learning performance against the baseline on the same task distribution.
*   **Models:** `conv2`, `conv4`, `conv6`.
*   **Data:** PB Shapes (`data/meta_h5/`).
*   **Training:** Meta-Learning (episodic).
*   **Testing:** Same task types as training (in-distribution).
*   **Task:** Same-Different Classification.
*   **Hypothesis/Finding:** Meta-learning outperforms regular SGD. (Confirmed).
*   **Status:** Completed.

### 3. Holdout (`holdout_experiment/`)

*   **Goal:** Evaluate meta-learning generalization to unseen task types (out-of-distribution).
*   **Models:** `conv2`, `conv4`, `conv6`.
*   **Data:** PB Shapes (`data/meta_h5/`).
*   **Training:** Meta-Learning (on a subset of tasks).
*   **Testing:** Held-out task types not seen during training.
*   **Task:** Same-Different Classification.
*   **Hypothesis/Finding:** Outperforms baseline SGD, but performs slightly worse than meta-baseline due to OOD generalization challenge. (Confirmed).
*   **Status:** Completed.

### 4. Naturalistic (`naturalistic/`)

*   **Goal:** Evaluate and compare standard SGD vs. meta-learning on naturalistic image data.
*   **Models:** `conv2`, `conv4`, `conv6`.
*   **Data:** Naturalistic Images (`data/meta_h5/`, `data/vanilla_h5/`).
*   **Training:** Regular SGD vs. Meta-Learning.
*   **Task:** Same-Different Classification.
*   **Hypothesis/Finding:** Meta-learning expected to outperform SGD. (Partially Confirmed - PB-trained models tested on naturalistic data showed meta-learning advantage).
*   **Status:** In Progress.
    *   **Next Steps:** Train models *on* naturalistic data and test them on naturalistic data.

### 5. Variable Tasks (`variable_tasks/` - *Folder name inferred, needs confirmation*)

*   **Goal:** Analyze how the number of training tasks affects meta-learning performance.
*   **Models:** `conv2`, `conv4`, `conv6`.
*   **Data:** PB Shapes (Assumed, `data/meta_h5/`).
*   **Training:** Meta-Learning (varying number of tasks).
*   **Task:** Same-Different Classification.
*   **Hypothesis:** Performance improves with more training tasks. (Untested).
*   **Status:** Planned/Not Started.

### 6. Weight Space Analysis (`weight_space_analysis/`)

*   **Goal:** Analyze and visualize the learned weight spaces of meta-learned vs. vanilla-learned models using techniques like PCA.
*   **Models:** Weights from relevant experiments (Baselines, Meta Baseline, etc.).
*   **Method:** PCA, other visualization techniques.
*   **Status:** In Progress (Focus on improving visualizations).

## New Goals and Tasks

### Goal 2: Analyze Weight Spaces of Pre-trained vs. Meta-trained Models
Description: Compare the learned weight spaces of models trained with conventional supervised learning on single tasks versus those trained with MAML.
- [X] Implement PCA and t-SNE analysis on the weights of `conv` models.

## New Tasks

- [X] Create a single Slurm array script generator in `run_variable_task_experiment.py` to handle all seed/task combinations.
- [X] Add a `--test` flag to `run_variable_task_experiment.py` for safe, single-job dry runs.
- [X] Resolve all runtime errors (`ModuleNotFoundError`, `TypeError`) identified during testing on Della.
- [X] Create an aggregation and plotting script `variable_task/aggregate_and_plot_results.py` to analyze the results after the main experiment completes. 