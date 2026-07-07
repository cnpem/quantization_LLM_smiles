"""
CSV Generation Module
======================

Handles:
1. Converting raw TXT files (LLM output) into structured CSVs
2. Processing all CSVs for a task: validating SMILES, cross-referencing
   with metrics, and generating consolidated result files.
"""

import json
import os
from pathlib import Path

import pandas as pd

from src.validation import validate_smiles, parse_model_name


def convert_txt_to_csv(txt_dir: Path, csv_dir: Path) -> int:
    """
    Convert all .txt files in a directory to .csv files with a 'SMILES' column.

    Each line in the TXT file is treated as one SMILES string.
    Empty lines are skipped.

    Args:
        txt_dir: Path to directory containing raw .txt files.
        csv_dir: Path to output directory for .csv files.

    Returns:
        Number of files converted.
    """
    csv_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for txt_file in sorted(txt_dir.glob("*.txt")):
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read()
            lines = [line for line in content.split("\n") if line.strip()]

        df = pd.DataFrame(lines, columns=["SMILES"])
        csv_path = csv_dir / txt_file.with_suffix(".csv").name
        df.to_csv(csv_path, index=False)
        count += 1

    return count


def generate_results_csv(
    csv_dir: Path,
    metrics_json: Path,
    output_dir: Path,
    task_name: str,
    exclude_models: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process all CSV files for a task, validate SMILES, cross-reference with
    metrics, and generate two output CSVs:

    1. results_per_replica.csv — one row per replica with detailed metrics
    2. results_summary.csv — aggregated by model+quantization (mean ± std)

    Args:
        csv_dir: Directory containing per-replica CSV files.
        metrics_json: Path to the all_metrics.json file.
        output_dir: Directory to write the output CSVs.
        task_name: Name of the task (e.g., "task_1").
        exclude_models: List of model substrings to exclude.

    Returns:
        Tuple of (df_per_replica, df_summary) DataFrames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    exclude_models = exclude_models or []

    # Load metrics JSON
    with open(metrics_json, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    df_metrics = pd.DataFrame(metrics_data)

    # Process each CSV file
    rows = []
    for csv_file in sorted(csv_dir.glob("*.csv")):
        replica_name = csv_file.stem  # e.g., "gemma-4-31B-Q8_replica_8"

        # Skip if file doesn't match the expected pattern
        if "_replica_" not in replica_name:
            continue

        # Extract model and quantization
        model_quant = replica_name.split("_replica_")[0]
        model_base, quantization = parse_model_name(model_quant)

        # Skip excluded models
        if any(excl.lower() in model_base.lower() for excl in exclude_models):
            continue

        # Read and validate SMILES
        try:
            df_smiles = pd.read_csv(csv_file)
        except pd.errors.EmptyDataError:
            continue

        if "SMILES" not in df_smiles.columns or df_smiles.empty:
            continue

        smiles_list = df_smiles["SMILES"].dropna().tolist()
        valid_smiles = validate_smiles(smiles_list)
        duplicates = df_smiles.duplicated(subset=["SMILES"]).sum()

        total = len(smiles_list)
        valid_count = len(valid_smiles)
        invalid_count = total - valid_count

        pct_valid = (valid_count / total * 100) if total > 0 else 0.0
        pct_invalid = (invalid_count / total * 100) if total > 0 else 0.0

        # Cross-reference with metrics for this model
        df_model_metrics = df_metrics[df_metrics["model"] == model_quant].copy()

        duration_min_mean = 0.0
        token_per_second_mean = 0.0

        if not df_model_metrics.empty:
            df_model_metrics["duration_min"] = (
                df_model_metrics["duration_ms"] / 60000.0
            )
            df_model_metrics["token_per_second"] = df_model_metrics["tokens"].apply(
                lambda x: x["tokens_per_second"] if isinstance(x, dict) else 0.0
            )
            duration_min_mean = df_model_metrics["duration_min"].mean()
            token_per_second_mean = df_model_metrics["token_per_second"].mean()

        rows.append(
            {
                "task": task_name,
                "modelo": model_base,
                "quantizacao": quantization,
                "replica": replica_name,
                "total_smiles": total,
                "total_smiles_validos": valid_count,
                "total_smiles_duplicados": duplicates,
                "pct_validos": pct_valid,
                "pct_invalidos": pct_invalid,
                "duration_min_media": duration_min_mean,
                "token_per_second_medio": token_per_second_mean,
            }
        )

    # Build per-replica DataFrame
    df_per_replica = pd.DataFrame(rows)

    if df_per_replica.empty:
        print(f"  ⚠️  No valid CSVs found in {csv_dir}")
        return df_per_replica, pd.DataFrame()

    # Save per-replica CSV
    per_replica_path = output_dir / "results_per_replica.csv"
    df_per_replica.to_csv(per_replica_path, index=False)
    print(f"  ✅ Per-replica results: {per_replica_path}")

    # Build summary DataFrame (aggregated by model+quantization)
    numeric_cols = [
        "total_smiles",
        "total_smiles_validos",
        "total_smiles_duplicados",
        "pct_validos",
        "pct_invalidos",
        "duration_min_media",
        "token_per_second_medio",
    ]

    df_summary_mean = (
        df_per_replica.groupby(["modelo", "quantizacao"])[numeric_cols]
        .mean()
        .reset_index()
    )
    df_summary_std = (
        df_per_replica.groupby(["modelo", "quantizacao"])[numeric_cols]
        .std()
        .reset_index()
        .fillna(0)
    )

    df_summary = pd.merge(
        df_summary_mean,
        df_summary_std,
        on=["modelo", "quantizacao"],
        suffixes=("_mean", "_std"),
    )

    # Save summary CSV
    summary_path = output_dir / "results_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"  ✅ Summary results:    {summary_path}")

    return df_per_replica, df_summary
