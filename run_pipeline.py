#!/usr/bin/env python3
"""
GCRM Pipeline Runner
=====================

Single entry point to reproduce all results from the paper.

Usage:
    python run_pipeline.py                  # Process all tasks
    python run_pipeline.py --task task_1    # Process only task_1
    python run_pipeline.py --skip-convert   # Skip TXT→CSV conversion
    python run_pipeline.py --skip-plots     # Skip plot generation
"""

import argparse
import sys
import time

from config import (
    TASKS,
    COLOR_MAP,
    QUANTIZATIONS,
    EXCLUDE_MODELS,
    FIGURE_DPI,
)
from src.csv_generator import convert_txt_to_csv, generate_results_csv
from src.plotting import generate_all_plots


def process_task(task_name: str, task_config: dict, skip_convert: bool, skip_plots: bool):
    """Process a single task: convert → validate → plot."""
    print(f"\n{'='*60}")
    print(f"  Processing: {task_name}")
    print(f"{'='*60}")

    raw_dir = task_config["raw_dir"]
    csv_dir = task_config["csv_dir"]
    metrics_json = task_config["metrics_json"]
    output_dir = task_config["output_dir"]

    # Step 1: Convert TXT → CSV
    if not skip_convert:
        print(f"\n  [1/3] Converting TXT → CSV...")
        if raw_dir.exists():
            count = convert_txt_to_csv(raw_dir, csv_dir)
            print(f"  ✅ Converted {count} files → {csv_dir}")
        else:
            print(f"  ⚠️  Raw directory not found: {raw_dir}")
            print(f"       Using existing CSVs from: {csv_dir}")
    else:
        print(f"\n  [1/3] Skipping TXT→CSV conversion (--skip-convert)")

    # Step 2: Validate SMILES & Generate Results CSVs
    print(f"\n  [2/3] Validating SMILES & generating results...")
    if not csv_dir.exists():
        print(f"  ❌ CSV directory not found: {csv_dir}")
        return

    if not metrics_json.exists():
        print(f"  ❌ Metrics JSON not found: {metrics_json}")
        return

    df_per_replica, df_summary = generate_results_csv(
        csv_dir=csv_dir,
        metrics_json=metrics_json,
        output_dir=output_dir,
        task_name=task_name,
        exclude_models=EXCLUDE_MODELS,
    )

    if df_per_replica.empty:
        print(f"  ❌ No data to process for {task_name}")
        return

    # Step 3: Generate Plots
    if not skip_plots:
        print(f"\n  [3/3] Generating plots...")
        generate_all_plots(
            df_per_replica=df_per_replica,
            metrics_json=metrics_json,
            output_dir=output_dir,
            color_map=COLOR_MAP,
            quantizations=QUANTIZATIONS,
            exclude_models=EXCLUDE_MODELS,
            dpi=FIGURE_DPI,
        )
    else:
        print(f"\n  [3/3] Skipping plot generation (--skip-plots)")

    print(f"\n  ✅ {task_name} complete! Results in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="GCRM Pipeline — Reproduce paper results with one command."
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help="Process only a specific task (e.g., 'task_1'). Default: all tasks.",
    )
    parser.add_argument(
        "--skip-convert", action="store_true",
        help="Skip the TXT→CSV conversion step.",
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip plot generation.",
    )
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║          GCRM Pipeline — Paper Reproducibility          ║")
    print("╚══════════════════════════════════════════════════════════╝")

    start = time.time()

    # Determine which tasks to process
    if args.task:
        if args.task not in TASKS:
            print(f"\n❌ Task '{args.task}' not found. Available: {list(TASKS.keys())}")
            sys.exit(1)
        tasks_to_run = {args.task: TASKS[args.task]}
    else:
        tasks_to_run = TASKS

    # Process each task
    for task_name, task_config in tasks_to_run.items():
        process_task(task_name, task_config, args.skip_convert, args.skip_plots)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  All done! Total time: {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
