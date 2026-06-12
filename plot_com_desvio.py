import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Carregar os dados iniciais de controle
df = pd.read_csv("results_smiles/results_quarta_leva_2.csv")


def preparar_dados(df, per_leva=False):
    df_c = df.copy()
    df_c["modelo_quant"] = df_c["replica"].apply(
        lambda x: x.split("_replica_")[0]
    )
    df_c["modelo"] = df_c["modelo_quant"].apply(lambda x: x.rsplit("-", 1)[0])
    df_c["quantizacao"] = df_c["modelo_quant"].apply(
        lambda x: x.rsplit("-", 1)[1]
    )
    df_c = df_c[~df_c["modelo"].str.contains("Mistral3.2", case=False)]

    if per_leva:
        primeira_leva = df_c[df_c["leva"] == "primeira_leva"]
        segunda_leva = df_c[df_c["leva"] == "segunda_leva"]
        return primeira_leva, segunda_leva
    else:
        return df_c


def plot_mean_com_desvio(leva):
    df_copy = df.copy()

    print(df_copy)

    # Tratamento dos nomes dos modelos
    df_copy["modelo_quant"] = df_copy["replica"].apply(
        lambda x: x.split("_replica_")[0]
    )
    df_copy["modelo"] = df_copy["modelo_quant"].apply(
        lambda x: x.rsplit("-", 1)[0]
    )
    df_copy["quantizacao"] = df_copy["modelo_quant"].apply(
        lambda x: x.rsplit("-", 1)[1]
    )
    df_copy = df_copy[~df_copy["modelo"].str.contains("Mistral3.2", case=False)]

    df_copy["total_smiles_invalidos"] = df_copy["total_smiles"] - df_copy["total_smiles_validos"]

    # Agrupamentos de Média e Desvio Padrão
    df_mean = (
        df_copy.groupby(["modelo", "quantizacao"])[
            ["total_smiles", "total_smiles_validos", "total_smiles_duplicados", "total_smiles_invalidos"]
        ]
        .mean()
        .reset_index()
    )
    df_std = (
        df_copy.groupby(["modelo", "quantizacao"])[
            ["total_smiles", "total_smiles_validos", "total_smiles_duplicados", "total_smiles_invalidos"]
        ]
        .std()
        .reset_index()
    )
    df_grouped = pd.merge(
        df_mean, df_std, on=["modelo", "quantizacao"], suffixes=("", "_std")
    ).fillna(0)

    print(df_grouped)

    modelos = sorted(df_grouped["modelo"].unique())
    quantizacao = ["BF16", "Q8", "Q6", "Q4"]

    x = np.arange(len(modelos))
    largura = 0.8 / len(quantizacao)

    # Configuração dos plots independentes: coluna_df -> (Título, Cor Base, Prefixo do Arquivo)
    config_plots = {
        "total_smiles": ("Total Generated SMILES", "#4c72b0", "total"),
        "total_smiles_validos": ("Valid SMILES", "#55a868", "validos"),
        "total_smiles_duplicados": ("Duplicate SMILES", "#4e5de9", "duplicados"),
        "total_smiles_invalidos": ("Invalid SMILES", "#ff0000", "invalidos")
    }

    # Loop principal: criará um arquivo diferente para cada métrica
    for coluna, (titulo, cor, prefixo_arquivo) in config_plots.items():

        # 1. Cria uma nova figura do zero para este gráfico específico
        fig, ax = plt.subplots(figsize=(14, 8))

        for i, quant in enumerate(quantizacao):
            df_q = df_grouped[(df_grouped["quantizacao"] == quant)]
            df_plot = (
                pd.DataFrame({"modelo": modelos})
                .merge(df_q, on="modelo", how="left")
                .fillna(0)
            )

            deslocamento = (i - len(quantizacao) / 2) * largura + largura / 2

            # Plot da barra com o desvio padrão correto no topo
            ax.bar(
                x + deslocamento,
                df_plot[coluna],
                largura,
                color=cor,
                alpha=1.0 - (i * 0.25),
                edgecolor="white",
                yerr=df_plot[f"{coluna}_std"],
                ecolor="black",
                capsize=3,
                error_kw={"elinewidth": 1.2},
            )

            # Rótulo de texto acima do desvio padrão
            for pos_x, altura_y, desvio_y in zip(
                x + deslocamento, df_plot[coluna], df_plot[f"{coluna}_std"]
            ):
                if altura_y == 0:
                    continue
                posicao_y = altura_y + desvio_y + 0.4
                ax.text(
                    pos_x,
                    posicao_y,
                    quant,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    rotation=45,
                )

        # 2. Configurações estéticas do gráfico atual
        ax.grid(axis="y", linestyle="--", alpha=0.2)
        ax.set_title(titulo, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Models", fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean SMILES per Replica", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(modelos, rotation=15, ha="right", fontweight="bold")

        # Força uma escala fixa de Y de 0 a 24 para manter a proporção visual correta entre as imagens
        ax.set_ylim(0, 24)

        plt.tight_layout()

        # 3. Salva cada gráfico com um nome diferente baseado no prefixo definido
        nome_saida = f"./grafico_separado_{prefixo_arquivo}.png"
        plt.savefig(nome_saida, dpi=150, bbox_inches="tight")
        print(f"Gráfico salvo com sucesso: {nome_saida}")

        # 4. Exibe na tela e FECHA a figura atual para limpar o painel do Matplotlib
        plt.show()
        plt.close(fig)


def plot_tempo_com_desvio(json_duration):
    with open(json_duration, "r", encoding="utf-8") as f:
        dados = json.load(f)

    df = pd.DataFrame(dados)

    df["modelo"] = df["model"].apply(lambda x: "-".join(x.split("-")[:-1]))
    df["quantizacao"] = df["model"].apply(lambda x: x.split("-")[-1])
    df = df[~df["modelo"].str.contains("Mistral3.2", case=False)]
    df["duration_min"] = df["duration_ms"] / 60000.0

    df_mean_d = (
        df.groupby(["modelo", "quantizacao"])["duration_min"]
        .mean()
        .reset_index()
    )
    df_std_d = (
        df.groupby(["modelo", "quantizacao"])["duration_min"]
        .std()
        .reset_index()
        .fillna(0)
    )
    df_grouped = pd.merge(
        df_mean_d, df_std_d, on=["modelo", "quantizacao"], suffixes=("", "_std")
    )

    modelos = sorted(df_grouped["modelo"].unique())
    quantizacoes = ["BF16", "Q8", "Q6", "Q4"]

    x = np.arange(len(modelos))
    largura = 0.8 / len(quantizacoes)

    fig, ax = plt.subplots(figsize=(14, 8))

    cor_bf16 = "#55a868"
    cor_q8 = "#4e5de9"
    cor_q6 = "#ff0000"
    cor_q4 = "#ffe600"
    cor_map = {"BF16": cor_bf16, "Q8": cor_q8, "Q6": cor_q6, "Q4": cor_q4}

    max_val = (df_grouped["duration_min"] + df_grouped["duration_min_std"]).max()

    for i, quant in enumerate(quantizacoes):
        df_q = df_grouped[df_grouped["quantizacao"] == quant]
        df_plot = (
            pd.DataFrame({"modelo": modelos})
            .merge(df_q, on="modelo", how="left")
            .fillna(0)
        )

        deslocamento = (i - len(quantizacoes) / 2) * largura + largura / 2

        ax.bar(
            x + deslocamento,
            df_plot["duration_min"],
            largura,
            color=cor_map[quant],
            edgecolor="white",
            yerr=df_plot["duration_min_std"],
            ecolor="black",
            capsize=3,
            error_kw={"elinewidth": 1.2},
            label=quant,
        )

        for pos_x, altura_y, desvio_y in zip(
            x + deslocamento,
            df_plot["duration_min"],
            df_plot["duration_min_std"],
        ):
            if altura_y == 0:
                continue
            posicao_y = altura_y + desvio_y + (max_val * 0.01)
            ax.text(
                pos_x,
                posicao_y,
                quant,
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                rotation=45,
            )

    ax.grid(axis="y", linestyle="--", alpha=0.2)

    y_step = 5
    teto_grafico = np.ceil(max_val / y_step) * y_step + y_step
    ax.set_yticks(np.arange(0, teto_grafico, y_step))
    ax.set_ylim(0, teto_grafico)

    ax.set_xlabel("Models", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel(
        "Mean Response Time (minutes)", fontsize=12, fontweight="bold"
    )
    ax.set_title(
        "Mean Response Time per Model and Quantization (in Minutes)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(modelos, rotation=15, ha="right", fontsize=11)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", title="Quantization")

    plt.tight_layout()
    plt.savefig("./tempo_duracao_min.png", dpi=150, bbox_inches="tight")
    print("Gráfico salvo com sucesso: ./tempo_duracao_min.png")
    plt.show()
    plt.close(fig)


def plot_tokens_com_desvio(json_duration):
    with open(json_duration, "r", encoding="utf-8") as f:
        dados = json.load(f)

    df = pd.DataFrame(dados)

    df["modelo"] = df["model"].apply(lambda x: "-".join(x.split("-")[:-1]))
    df["quantizacao"] = df["model"].apply(lambda x: x.split("-")[-1])
    df["token_per_second"] = df["tokens"].apply(
        lambda x: x["tokens_per_second"]
    )
    df = df[~df["modelo"].str.contains("Mistral3.2", case=False)]

    df_seg_m = (
        df.groupby(["modelo", "quantizacao"])["token_per_second"]
        .mean()
        .reset_index()
    )
    df_seg_std = (
        df.groupby(["modelo", "quantizacao"])["token_per_second"]
        .std()
        .reset_index()
        .fillna(0)
    )
    df_grouped = pd.merge(
        df_seg_m, df_seg_std, on=["modelo", "quantizacao"], suffixes=("", "_std")
    )

    modelos = sorted(df_grouped["modelo"].unique())
    quantizacoes = ["BF16", "Q8", "Q6", "Q4"]

    x = np.arange(len(modelos))
    largura = 0.8 / len(quantizacoes)

    fig, ax = plt.subplots(figsize=(14, 8))

    cor_bf16 = "#55a868"
    cor_q8 = "#4e5de9"
    cor_q6 = "#ff0000"
    cor_q4 = "#ffe600"
    cor_map = {"BF16": cor_bf16, "Q8": cor_q8, "Q6": cor_q6, "Q4": cor_q4}

    max_val = (
        df_grouped["token_per_second"] + df_grouped["token_per_second_std"]
    ).max()

    for i, quant in enumerate(quantizacoes):
        df_q = df_grouped[df_grouped["quantizacao"] == quant]
        df_plot = (
            pd.DataFrame({"modelo": modelos})
            .merge(df_q, on="modelo", how="left")
            .fillna(0)
        )

        deslocamento = (i - len(quantizacoes) / 2) * largura + largura / 2

        ax.bar(
            x + deslocamento,
            df_plot["token_per_second"],
            largura,
            color=cor_map[quant],
            edgecolor="white",
            yerr=df_plot["token_per_second_std"],
            ecolor="black",
            capsize=3,
            error_kw={"elinewidth": 1.2},
            label=quant,
        )

        for pos_x, altura_y, desvio_y in zip(
            x + deslocamento,
            df_plot["token_per_second"],
            df_plot["token_per_second_std"],
        ):
            if altura_y == 0:
                continue
            posicao_y = altura_y + desvio_y + (max_val * 0.01)
            ax.text(
                pos_x,
                posicao_y,
                quant,
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                rotation=45,
            )

    ax.grid(axis="y", linestyle="--", alpha=0.2)

    y_step = 5
    teto_grafico = np.ceil(max_val / y_step) * y_step + y_step
    ax.set_yticks(np.arange(0, teto_grafico, y_step))
    ax.set_ylim(0, teto_grafico)

    ax.set_xlabel("Models", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel(
        "Mean Tokens per Second (tokens/s)", fontsize=12, fontweight="bold"
    )
    ax.set_title(
        "Mean Tokens per Second per Model and Quantization",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(modelos, rotation=15, ha="right", fontsize=11)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", title="Quantization")

    plt.tight_layout()
    plt.savefig("tokens_seg_com_desvio.png", dpi=150, bbox_inches="tight")
    print("Gráfico salvo com sucesso: tokens_seg_com_desvio.png")
    plt.show()
    plt.close(fig)


def plot_porcentagem_com_desvio():
    """
    Plota a porcentagem média (± desvio padrão) de SMILES válidos e inválidos
    por modelo e quantização, usando os dados por réplica de results_quarta_leva_2.csv.
    """
    df_pct = pd.read_csv("results_smiles/results_quarta_leva_2.csv")

    # Tratamento dos nomes dos modelos
    df_pct["modelo_quant"] = df_pct["replica"].apply(
        lambda x: x.split("_replica_")[0]
    )
    df_pct["modelo"] = df_pct["modelo_quant"].apply(
        lambda x: x.rsplit("-", 1)[0]
    )
    df_pct["quantizacao"] = df_pct["modelo_quant"].apply(
        lambda x: x.rsplit("-", 1)[1]
    )
    df_pct = df_pct[~df_pct["modelo"].str.contains("Mistral3.2", case=False)]

    # Calcular porcentagens por réplica (caso as colunas não existam no CSV)
    if "pct_validos" not in df_pct.columns:
        df_pct["pct_validos"] = (
            df_pct["total_smiles_validos"] / df_pct["total_smiles"]
        ) * 100
    if "pct_invalidos" not in df_pct.columns:
        df_pct["pct_invalidos"] = (
            (df_pct["total_smiles"] - df_pct["total_smiles_validos"])
            / df_pct["total_smiles"]
        ) * 100

    # Média e desvio padrão das porcentagens por modelo+quantização
    cols_pct = ["pct_validos", "pct_invalidos"]
    df_mean = (
        df_pct.groupby(["modelo", "quantizacao"])[cols_pct]
        .mean()
        .reset_index()
    )
    df_std = (
        df_pct.groupby(["modelo", "quantizacao"])[cols_pct]
        .std()
        .reset_index()
        .fillna(0)
    )
    df_grouped = pd.merge(
        df_mean, df_std, on=["modelo", "quantizacao"], suffixes=("", "_std")
    )

    print("\n--- Dados de porcentagem (média ± std) ---")
    print(df_grouped)

    modelos = sorted(df_grouped["modelo"].unique())
    quantizacoes = ["BF16", "Q8", "Q6", "Q4"]

    x = np.arange(len(modelos))
    largura = 0.8 / len(quantizacoes)

    cor_bf16 = "#55a868"
    cor_q8 = "#4e5de9"
    cor_q6 = "#ff0000"
    cor_q4 = "#ffe600"
    cor_map = {"BF16": cor_bf16, "Q8": cor_q8, "Q6": cor_q6, "Q4": cor_q4}

    # Configuração dos plots de porcentagem
    config_plots_pct = {
        "pct_validos": ("% Valid SMILES", "#55a868", "pct_validos"),
        "pct_invalidos": ("% Invalid SMILES", "#ff0000", "pct_invalidos"),
    }

    for coluna, (titulo, cor, prefixo_arquivo) in config_plots_pct.items():
        fig, ax = plt.subplots(figsize=(14, 8))

        for i, quant in enumerate(quantizacoes):
            df_q = df_grouped[df_grouped["quantizacao"] == quant]
            df_plot = (
                pd.DataFrame({"modelo": modelos})
                .merge(df_q, on="modelo", how="left")
                .fillna(0)
            )

            deslocamento = (i - len(quantizacoes) / 2) * largura + largura / 2

            ax.bar(
                x + deslocamento,
                df_plot[coluna],
                largura,
                color=cor_map[quant],
                edgecolor="white",
                yerr=df_plot[f"{coluna}_std"],
                ecolor="black",
                capsize=3,
                error_kw={"elinewidth": 1.2},
                label=quant,
            )

            # Rótulo de texto acima do desvio padrão
            for pos_x, altura_y, desvio_y in zip(
                x + deslocamento, df_plot[coluna], df_plot[f"{coluna}_std"]
            ):
                if altura_y == 0:
                    continue
                posicao_y = altura_y + desvio_y + 0.5
                ax.text(
                    pos_x,
                    posicao_y,
                    quant,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    rotation=45,
                )

        ax.grid(axis="y", linestyle="--", alpha=0.2)
        ax.set_title(titulo, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Models", fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean Percentage (%)", fontsize=11, fontweight="bold")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", title="Quantization")
        ax.set_xticks(x)
        ax.set_xticklabels(modelos, rotation=15, ha="right", fontweight="bold")
        ax.set_ylim(0, 120)

        plt.tight_layout()

        nome_saida = f"./grafico_separado_{prefixo_arquivo}.png"
        plt.savefig(nome_saida, dpi=150, bbox_inches="tight")
        print(f"Gráfico salvo com sucesso: {nome_saida}")

        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    # plot_mean_com_desvio(
    #     "/home/lucas.abner/Documentos/code/gcrm/results_smiles/results_20_smiles_agregado.csv"
    # )
    plot_porcentagem_com_desvio()
    plot_tempo_com_desvio(
        "/home/lucas.abner/Documentos/code/gcrm/quarta_leva_menos_moleculas/all_metrics.json"
    )
    plot_tokens_com_desvio(
        "/home/lucas.abner/Documentos/code/gcrm/quarta_leva_menos_moleculas/all_metrics.json"
    )