"""
Plotting Module
================

All visualization functions for the GCRM paper.
Each function receives DataFrames and config parameters instead of reading files directly.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _prepare_plot_data(df_per_replica, exclude_models=None):
    """Prepare data for plotting: filter excluded models, compute stats."""
    df = df_per_replica.copy()
    exclude_models = exclude_models or []
    for excl in exclude_models:
        df = df[~df["modelo"].str.contains(excl, case=False)]

    df["total_smiles_invalidos"] = df["total_smiles"] - df["total_smiles_validos"]

    if "pct_validos" not in df.columns:
        df["pct_validos"] = (df["total_smiles_validos"] / df["total_smiles"]) * 100
    if "pct_invalidos" not in df.columns:
        df["pct_invalidos"] = (
            (df["total_smiles"] - df["total_smiles_validos"]) / df["total_smiles"]
        ) * 100

    return df


def _grouped_mean_std(df, group_cols, value_cols):
    """Compute mean and std grouped by group_cols."""
    df_mean = df.groupby(group_cols)[value_cols].mean().reset_index()
    df_std = df.groupby(group_cols)[value_cols].std().reset_index().fillna(0)
    return pd.merge(df_mean, df_std, on=group_cols, suffixes=("", "_std"))


def plot_smiles_bar_chart(
    df_per_replica, output_dir, color_map, quantizations, exclude_models=None,
    dpi=150, figsize=(14, 8),
):
    """
    Generate separate bar charts for each SMILES metric:
    Total Generated, Valid, Duplicate, Invalid — each with std dev error bars.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _prepare_plot_data(df_per_replica, exclude_models)
    value_cols = ["total_smiles", "total_smiles_validos", "total_smiles_duplicados", "total_smiles_invalidos"]
    df_grouped = _grouped_mean_std(df, ["modelo", "quantizacao"], value_cols)

    modelos = sorted(df_grouped["modelo"].unique())
    x = np.arange(len(modelos))
    largura = 0.8 / len(quantizations)

    config_plots = {
        "total_smiles": ("Total Generated SMILES", "total"),
        "total_smiles_validos": ("Valid SMILES", "validos"),
        "total_smiles_duplicados": ("Duplicate SMILES", "duplicados"),
        "total_smiles_invalidos": ("Invalid SMILES", "invalidos"),
    }

    for coluna, (titulo, prefixo) in config_plots.items():
        fig, ax = plt.subplots(figsize=figsize)

        for i, quant in enumerate(quantizations):
            df_q = df_grouped[df_grouped["quantizacao"] == quant]
            df_plot = pd.DataFrame({"modelo": modelos}).merge(df_q, on="modelo", how="left").fillna(0)
            desl = (i - len(quantizations) / 2) * largura + largura / 2

            ax.bar(
                x + desl, df_plot[coluna], largura,
                color=color_map.get(quant, "#999999"),
                edgecolor="white",
                yerr=df_plot[f"{coluna}_std"],
                ecolor="black", capsize=3, error_kw={"elinewidth": 1.2},
            )

            for px, hy, sy in zip(x + desl, df_plot[coluna], df_plot[f"{coluna}_std"]):
                if hy == 0:
                    continue
                ax.text(px, hy + sy + 0.4, quant, ha="center", va="bottom",
                        fontsize=11, fontweight="bold", rotation=45)

        ax.grid(axis="y", linestyle="--", alpha=0.2)
        ax.tick_params(axis="y", labelsize=13)
        ax.set_title(titulo, fontsize=17, fontweight="bold", pad=12)
        ax.set_xlabel("Models", fontsize=14, fontweight="bold")
        ax.set_ylabel("Mean SMILES per Replica", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(modelos, rotation=15, ha="right", fontweight="bold", fontsize=14)
        ax.set_ylim(0, 24)
        plt.tight_layout()

        path = output_dir / f"bar_{prefixo}.png"
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  📊 {path.name}")


def plot_percentage_combined(
    df_per_replica, output_dir, color_map, quantizations, exclude_models=None,
    dpi=150, figsize=(14, 14),
):
    """Plot % Valid and % Invalid SMILES as two vertically stacked subplots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _prepare_plot_data(df_per_replica, exclude_models)
    df_grouped = _grouped_mean_std(df, ["modelo", "quantizacao"], ["pct_validos", "pct_invalidos"])

    modelos = sorted(df_grouped["modelo"].unique())
    x = np.arange(len(modelos))
    largura = 0.8 / len(quantizations)

    config_subplots = [
        ("pct_validos", "% Valid SMILES"),
        ("pct_invalidos", "% Invalid SMILES"),
    ]

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize, sharex=True)

    for ax, (coluna, titulo) in zip(axes, config_subplots):
        for i, quant in enumerate(quantizations):
            df_q = df_grouped[df_grouped["quantizacao"] == quant]
            df_plot = pd.DataFrame({"modelo": modelos}).merge(df_q, on="modelo", how="left").fillna(0)
            desl = (i - len(quantizations) / 2) * largura + largura / 2

            ax.bar(
                x + desl, df_plot[coluna], largura,
                color=color_map.get(quant, "#999999"),
                edgecolor="white",
                yerr=df_plot[f"{coluna}_std"],
                ecolor="black", capsize=3, error_kw={"elinewidth": 1.2},
                label=quant,
            )

            for px, hy, sy in zip(x + desl, df_plot[coluna], df_plot[f"{coluna}_std"]):
                if hy == 0:
                    continue
                ax.text(px, hy + sy + 0.5, quant, ha="center", va="bottom",
                        fontsize=11, fontweight="bold", rotation=45)

        ax.grid(axis="y", linestyle="--", alpha=0.2)
        ax.tick_params(axis="y", labelsize=13)
        ax.set_title(titulo, fontsize=17, fontweight="bold", pad=12)
        ax.set_ylabel("Mean Percentage (%)", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 120)

    axes[-1].set_xlabel("Models", fontsize=14, fontweight="bold")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(modelos, rotation=15, ha="right", fontweight="bold", fontsize=14)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", title="Quantization",
               fontsize=14, title_fontsize=15, bbox_to_anchor=(0.99, 0.98))

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    path = output_dir / "pct_combined.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 {path.name}")


def _load_metrics_df(metrics_json, exclude_models=None):
    """Load and prepare metrics DataFrame from JSON."""
    with open(metrics_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["modelo"] = df["model"].apply(lambda x: "-".join(x.split("-")[:-1]))
    df["quantizacao"] = df["model"].apply(lambda x: x.split("-")[-1])

    exclude_models = exclude_models or []
    for excl in exclude_models:
        df = df[~df["modelo"].str.contains(excl, case=False)]

    df["duration_min"] = df["duration_ms"] / 60000.0
    df["token_per_second"] = df["tokens"].apply(
        lambda x: x["tokens_per_second"] if isinstance(x, dict) else 0.0
    )
    return df


def plot_response_time_bar(
    metrics_json, output_dir, color_map, quantizations, exclude_models=None,
    dpi=150, figsize=(14, 8),
):
    """Bar chart of mean response time (minutes) with std dev error bars."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_metrics_df(metrics_json, exclude_models)
    df_grouped = _grouped_mean_std(df, ["modelo", "quantizacao"], ["duration_min"])

    modelos = sorted(df_grouped["modelo"].unique())
    x = np.arange(len(modelos))
    largura = 0.8 / len(quantizations)

    fig, ax = plt.subplots(figsize=figsize)
    max_val = (df_grouped["duration_min"] + df_grouped["duration_min_std"]).max()

    for i, quant in enumerate(quantizations):
        df_q = df_grouped[df_grouped["quantizacao"] == quant]
        df_plot = pd.DataFrame({"modelo": modelos}).merge(df_q, on="modelo", how="left").fillna(0)
        desl = (i - len(quantizations) / 2) * largura + largura / 2

        ax.bar(
            x + desl, df_plot["duration_min"], largura,
            color=color_map.get(quant, "#999999"), edgecolor="white",
            yerr=df_plot["duration_min_std"],
            ecolor="black", capsize=3, error_kw={"elinewidth": 1.2}, label=quant,
        )

        for px, hy, sy in zip(x + desl, df_plot["duration_min"], df_plot["duration_min_std"]):
            if hy == 0:
                continue
            ax.text(px, hy + sy + (max_val * 0.01), quant,
                    ha="center", va="bottom", fontsize=11, fontweight="bold", rotation=45)

    ax.grid(axis="y", linestyle="--", alpha=0.2)
    ax.tick_params(axis="y", labelsize=13)
    y_step = 5
    teto = np.ceil(max_val / y_step) * y_step + y_step
    ax.set_yticks(np.arange(0, teto, y_step))
    ax.set_ylim(0, teto)
    ax.set_xlabel("Models", fontsize=15, fontweight="bold", labelpad=10)
    ax.set_ylabel("Mean Response Time (minutes)", fontsize=15, fontweight="bold")
    ax.set_title("Mean Response Time per Model and Quantization (in Minutes)",
                 fontsize=17, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(modelos, rotation=15, ha="right", fontsize=14)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", title="Quantization",
              fontsize=13, title_fontsize=14)
    plt.tight_layout()

    path = output_dir / "response_time_bar.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 {path.name}")


def plot_response_time_boxplot(
    metrics_json, output_dir, color_map, quantizations, exclude_models=None,
    dpi=150, figsize=(14, 8),
):
    """Boxplot of response time distribution with inset zoom."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_metrics_df(metrics_json, exclude_models)
    modelos = sorted(df["modelo"].unique())
    x = np.arange(len(modelos))
    largura = 0.8 / len(quantizations)

    fig, ax = plt.subplots(figsize=figsize)
    max_val = df["duration_min"].max()

    # Create inset axes for zoom
    ax_inset = ax.inset_axes([0.72, 0.62, 0.25, 0.32])

    for i, quant in enumerate(quantizations):
        desl = (i - len(quantizations) / 2) * largura + largura / 2
        dados_bp, pos_bp = [], []
        dados_inset, pos_inset = [], []

        for j, modelo in enumerate(modelos):
            subset = df[(df["modelo"] == modelo) & (df["quantizacao"] == quant)]["duration_min"].values
            if len(subset) > 0:
                dados_bp.append(subset)
                pos_bp.append(x[j] + desl)
                if modelo == "Qwen3.6-35B":
                    dados_inset.append(subset)
                    pos_inset.append(x[j] + desl)
                elif modelo == "gemma-4-31B" and quant != "BF16":
                    dados_inset.append(subset)
                    pos_inset.append(x[j] + desl)

        bp_props = dict(
            widths=largura * 0.8, patch_artist=True, showfliers=True,
            medianprops={"color": "black", "linewidth": 1.2},
            boxprops={"edgecolor": "black", "linewidth": 1.0},
            whiskerprops={"color": "black", "linewidth": 1.0},
            capprops={"color": "black", "linewidth": 1.0},
            flierprops={"marker": "o", "markerfacecolor": "black",
                        "markeredgecolor": "black", "alpha": 0.4, "markersize": 4},
        )

        if dados_bp:
            bp = ax.boxplot(dados_bp, positions=pos_bp, **bp_props)
            for patch in bp["boxes"]:
                patch.set_facecolor(color_map.get(quant, "#999999"))

        if dados_inset:
            bp_props_inset = {**bp_props}
            bp_props_inset["medianprops"] = {"color": "black", "linewidth": 1.0}
            bp_props_inset["boxprops"] = {"edgecolor": "black", "linewidth": 0.8}
            bp_props_inset["whiskerprops"] = {"color": "black", "linewidth": 0.8}
            bp_props_inset["capprops"] = {"color": "black", "linewidth": 0.8}
            bp_props_inset["flierprops"] = {"marker": "o", "markerfacecolor": "black",
                                            "markeredgecolor": "black", "alpha": 0.3, "markersize": 3}
            bp_i = ax_inset.boxplot(dados_inset, positions=pos_inset, **bp_props_inset)
            for patch in bp_i["boxes"]:
                patch.set_facecolor(color_map.get(quant, "#999999"))

    ax.grid(axis="y", linestyle="--", alpha=0.2)
    ax.tick_params(axis="y", labelsize=13)
    y_step = 5 if max_val <= 30 else 10
    teto = np.ceil(max_val / y_step) * y_step + y_step
    ax.set_yticks(np.arange(0, teto, y_step))
    ax.set_ylim(0, teto)
    ax.set_xlabel("Models", fontsize=15, fontweight="bold", labelpad=10)
    ax.set_ylabel("Response Time (minutes)", fontsize=15, fontweight="bold")
    ax.set_title("Response Time Distribution per Model and Quantization (in Minutes)",
                 fontsize=17, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(modelos, rotation=15, ha="right", fontsize=14)

    # Inset config
    ax_inset.set_ylim(0, 15.0)
    ax_inset.set_xlim(0.5, 2.5)
    ax_inset.set_xticks([1, 2])
    ax_inset.set_xticklabels(["Qwen3.6-35B", "gemma-4-31B"], rotation=10, ha="right", fontsize=8)
    ax_inset.tick_params(axis="y", labelsize=10)
    ax_inset.grid(axis="y", linestyle="--", alpha=0.15)
    ax_inset.set_title("Zoom (0 - 15 min)", fontsize=10, fontweight="bold", pad=5)
    try:
        ax.indicate_inset_zoom(ax_inset, edgecolor="gray", alpha=0.3)
    except Exception:
        pass

    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color_map.get(q, "#999"), edgecolor="black", label=q)
        for q in quantizations
    ]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.01, 1), loc="upper left",
              title="Quantization", fontsize=13, title_fontsize=14)
    plt.tight_layout()

    path = output_dir / "response_time_boxplot.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 {path.name}")


def plot_tokens_per_second(
    metrics_json, output_dir, color_map, quantizations, exclude_models=None,
    dpi=150, figsize=(14, 8),
):
    """Bar chart of mean tokens/second with std dev error bars."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_metrics_df(metrics_json, exclude_models)
    df_grouped = _grouped_mean_std(df, ["modelo", "quantizacao"], ["token_per_second"])

    modelos = sorted(df_grouped["modelo"].unique())
    x = np.arange(len(modelos))
    largura = 0.8 / len(quantizations)

    fig, ax = plt.subplots(figsize=figsize)
    max_val = (df_grouped["token_per_second"] + df_grouped["token_per_second_std"]).max()

    for i, quant in enumerate(quantizations):
        df_q = df_grouped[df_grouped["quantizacao"] == quant]
        df_plot = pd.DataFrame({"modelo": modelos}).merge(df_q, on="modelo", how="left").fillna(0)
        desl = (i - len(quantizations) / 2) * largura + largura / 2

        ax.bar(
            x + desl, df_plot["token_per_second"], largura,
            color=color_map.get(quant, "#999999"), edgecolor="white",
            yerr=df_plot["token_per_second_std"],
            ecolor="black", capsize=3, error_kw={"elinewidth": 1.2}, label=quant,
        )

        for px, hy, sy in zip(x + desl, df_plot["token_per_second"], df_plot["token_per_second_std"]):
            if hy == 0:
                continue
            ax.text(px, hy + sy + (max_val * 0.01), quant,
                    ha="center", va="bottom", fontsize=11, fontweight="bold", rotation=45)

    ax.grid(axis="y", linestyle="--", alpha=0.2)
    ax.tick_params(axis="y", labelsize=13)
    y_step = 5
    teto = np.ceil(max_val / y_step) * y_step + y_step
    ax.set_yticks(np.arange(0, teto, y_step))
    ax.set_ylim(0, teto)
    ax.set_xlabel("Models", fontsize=15, fontweight="bold", labelpad=10)
    ax.set_ylabel("Mean Tokens per Second (tokens/s)", fontsize=15, fontweight="bold")
    ax.set_title("Mean Tokens per Second per Model and Quantization",
                 fontsize=17, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(modelos, rotation=15, ha="right", fontsize=14)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", title="Quantization",
              fontsize=13, title_fontsize=14)
    plt.tight_layout()

    path = output_dir / "tokens_per_second.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 {path.name}")


def plot_stacked_validation(
    df_per_replica, output_dir, color_map, quantizations, exclude_models=None,
    dpi=150, figsize=(14, 8),
):
    """Stacked bar chart: Unique Valid + Duplicates + Invalid per model/quantization."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _prepare_plot_data(df_per_replica, exclude_models)
    value_cols = ["total_smiles", "total_smiles_validos", "total_smiles_duplicados"]
    df_grouped = _grouped_mean_std(df, ["modelo", "quantizacao"], value_cols)

    modelos = sorted(df_grouped["modelo"].unique())
    x = np.arange(len(modelos))
    largura = 0.8 / len(quantizations)

    cor_valido = "#55a868"
    cor_duplicado = "#4e5de9"
    cor_invalido = "#ff0000"

    fig, ax = plt.subplots(figsize=figsize)

    for i, quant in enumerate(quantizations):
        df_q = df_grouped[df_grouped["quantizacao"] == quant]
        df_plot = pd.DataFrame({"modelo": modelos}).merge(df_q, on="modelo", how="left").fillna(0)
        desl = (i - len(quantizations) / 2) * largura + largura / 2

        validos_unicos = df_plot["total_smiles_validos"] - df_plot["total_smiles_duplicados"]
        invalidos = df_plot["total_smiles"] - df_plot["total_smiles_validos"]

        ax.bar(x + desl, validos_unicos, largura, color=cor_valido, edgecolor="white")
        ax.bar(x + desl, df_plot["total_smiles_duplicados"], largura,
               bottom=validos_unicos, color=cor_duplicado, edgecolor="white")
        ax.bar(x + desl, invalidos, largura,
               bottom=df_plot["total_smiles_validos"], color=cor_invalido, edgecolor="white")

        for px, hy in zip(x + desl, df_plot["total_smiles"]):
            if hy == 0:
                continue
            ax.text(px, hy + 0.3, quant, ha="center", va="bottom",
                    fontsize=12, fontweight="bold", rotation=60)

    ax.grid(axis="y", linestyle="--", alpha=0.2)
    ax.tick_params(axis="y", labelsize=13)

    max_h = df_grouped["total_smiles"].max()
    y_step = 5
    teto = int(np.ceil(max_h / float(y_step)) * y_step) + y_step
    ax.set_yticks(np.arange(0, teto, y_step))
    ax.set_ylim(0, teto)

    ax.set_xlabel("Models", fontsize=15, fontweight="bold", labelpad=10)
    ax.set_ylabel("Mean SMILES per Replica", fontsize=15, fontweight="bold")
    ax.set_title("SMILES Validation per Model and Quantization (Mean)",
                 fontsize=17, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(modelos, rotation=15, ha="right", fontsize=14)
    ax.legend(["Unique Valid", "Duplicates", "Invalid"],
              bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=13)
    plt.tight_layout()

    path = output_dir / "stacked_validation.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 {path.name}")


def generate_all_plots(df_per_replica, metrics_json, output_dir, color_map,
                       quantizations, exclude_models=None, dpi=150):
    """Generate all plots for a given task."""
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    common = dict(color_map=color_map, quantizations=quantizations,
                  exclude_models=exclude_models, dpi=dpi)

    plot_smiles_bar_chart(df_per_replica, plots_dir, **common)
    plot_percentage_combined(df_per_replica, plots_dir, **common, figsize=(14, 14))
    plot_stacked_validation(df_per_replica, plots_dir, **common)
    plot_response_time_bar(metrics_json, plots_dir, **common)
    plot_response_time_boxplot(metrics_json, plots_dir, **common)
    plot_tokens_per_second(metrics_json, plots_dir, **common)
