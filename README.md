# GCRM — Generative Chemistry Reproducibility Module

Pipeline for evaluating LLM performance on chemical SMILES generation tasks.
This code supports the reproducibility of results presented in our paper.

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Run the full pipeline (validation + CSV generation + plots)
uv run python run_pipeline.py
```

That's it! All results will be generated in the `results/` directory.

## How to Customize

Edit **`config.py`** to change:

| Setting | Description |
|---------|-------------|
| `TASKS` | Add/remove tasks, change data paths |
| `MODELS` | List of models to evaluate |
| `QUANTIZATIONS` | Quantization levels (`BF16`, `Q8`, `Q6`, `Q4`) |
| `EXCLUDE_MODELS` | Models to exclude from plots |
| `PROMPT_TASK_1/2` | The prompts sent to the LLMs |
| `COLOR_MAP` | Colors for each quantization in plots |
| `N_REPLICAS` | Number of replicas per model |

## Pipeline Options

```bash
# Process only a specific task
uv run python run_pipeline.py --task task_1

# Skip TXT→CSV conversion (if CSVs already exist)
uv run python run_pipeline.py --skip-convert

# Skip plot generation (only generate CSVs)
uv run python run_pipeline.py --skip-plots
```

## Project Structure

```
gcrm/
├── config.py              # ← Edit this to customize experiments
├── run_pipeline.py        # ← Run this to reproduce results
├── src/
│   ├── validation.py      # SMILES validation (RDKit)
│   ├── csv_generator.py   # Data processing & CSV generation
│   └── plotting.py        # All paper visualizations
├── task_1/                # Raw LLM outputs (Task 1)
├── task_2/                # Raw LLM outputs (Task 2)
├── csv_llms_task_1/       # Parsed CSVs (Task 1)
├── csv_llms_task_2/       # Parsed CSVs (Task 2)
├── results/               # Generated results (CSVs + plots)
│   ├── task_1/
│   │   ├── results_per_replica.csv
│   │   ├── results_summary.csv
│   │   └── plots/
│   └── task_2/
│       ├── results_per_replica.csv
│       ├── results_summary.csv
│       └── plots/
├── prompts.py             # Data collection script (requires LLM server)
└── pyproject.toml         # Project dependencies
```

## Output Files

For each task, the pipeline generates:

- **`results_per_replica.csv`** — One row per model/quantization/replica with:
  total SMILES, valid count, duplicates, percentages, response time, tokens/s
- **`results_summary.csv`** — Aggregated statistics (mean ± std) per model+quantization
- **`plots/`** — All paper figures (bar charts, boxplots, percentage plots)

## Requirements

- Python ≥ 3.12
- Dependencies: `matplotlib`, `numpy`, `pandas`, `rdkit`, `requests`