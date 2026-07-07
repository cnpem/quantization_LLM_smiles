import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# 1. Carregar os dados
df = pd.read_csv('results_smiles/results_task_2.csv')

def preparar_dados(df, per_leva=False):
    # 2. Extrair o Modelo e a Quantização da coluna 'replica'
    # A string tem o formato "Modelo-Quantizacao_replica_X" (ex: "Qwen3.6-35B-BF16_replica_4")
    if per_leva:
        df_c = df.copy()
        df_c['modelo_quant'] = df_c['replica'].apply(lambda x: x.split('_replica_')[0])
        # Pega tudo antes do último hífen como modelo
        df_c['modelo'] = df_c['modelo_quant'].apply(lambda x: x.rsplit('-', 1)[0])
        # Pega tudo depois do último hífen como quantização
        df_c['quantizacao'] = df_c['modelo_quant'].apply(lambda x: x.rsplit('-', 1)[1])
        df_c = df_c[~df_c["modelo"].str.contains("Mistral", case=False)]

        primeira_leva = df_c[df_c['leva'] == 'primeira_leva']
        segunda_leva = df_c[df_c['leva'] == 'segunda_leva']
        return primeira_leva, segunda_leva
    else:
        df_c = df.copy()
        df_c['modelo_quant'] = df_c['replica'].apply(lambda x: x.split('_replica_')[0])
        # Pega tudo antes do último hífen como modelo
        df_c['modelo'] = df_c['modelo_quant'].apply(lambda x: x.rsplit('-', 1)[0])
        # Pega tudo depois do último hífen como quantização
        df_c['quantizacao'] = df_c['modelo_quant'].apply(lambda x: x.rsplit('-', 1)[1])
        df_c = df_c[~df_c["replica"].str.contains("Mistral3.2", case=False)]

        return df_c

def plotar_grafico_media(leva, nome_arquivo):
    # 1. Agrupamos por MÉDIA para ter a altura média das fatias por réplica
    df_mean = leva.groupby(['modelo', 'quantizacao'])[['total_smiles', 'total_smiles_validos', 'total_smiles_duplicados']].mean().reset_index()
    
    # 2. Agrupamos por DESVIO PADRÃO (std) para medir a variação entre as réplicas
    df_std = leva.groupby(['modelo', 'quantizacao'])[['total_smiles_validos']].std().reset_index().fillna(0)
    
    # Combinamos a média e o desvio padrão
    df_grouped = pd.merge(df_mean, df_std, on=['modelo', 'quantizacao'], suffixes=('', '_std'))

    # Ordem fixa para as quantizações no eixo X
    modelos = sorted(df_grouped['modelo'].unique())
    quantizacoes = ['BF16', 'Q8', 'Q6', 'Q4']  

    x = np.arange(len(modelos))
    largura = 0.8 / len(quantizacoes)

    fig, ax = plt.subplots(figsize=(16, 8))

    cor_valido = "#55a868"     # Verde
    cor_duplicado = "#4e5de9"  # Azul
    cor_invalido = "#ff0000"   # Vermelho

    for i, quant in enumerate(quantizacoes):
        df_q = df_grouped[df_grouped['quantizacao'] == quant]
        df_plot = pd.DataFrame({'modelo': modelos}).merge(df_q, on='modelo', how='left').fillna(0)
        
        deslocamento = (i - len(quantizacoes)/2) * largura + largura/2
        
        # MATEMÁTICA DO EMPILHAMENTO (Baseado nas médias)
        validos_unicos = df_plot['total_smiles_validos'] - df_plot['total_smiles_duplicados']
        invalidos = df_plot['total_smiles'] - df_plot['total_smiles_validos']
        
        # PLOT 1: VÁLIDOS ÚNICOS (Base)
        ax.bar(x + deslocamento, validos_unicos, largura, color=cor_valido, edgecolor='white')

        # PLOT 2: DUPLICADOS (Meio)
        ax.bar(x + deslocamento, df_plot['total_smiles_duplicados'], largura, 
               bottom=validos_unicos, color=cor_duplicado, edgecolor='white')

        # PLOT 3: INVÁLIDOS (Topo) + DESVIO PADRÃO
        # O yerr continua funcionando exatamente igual, mas agora aplicado sobre a barra da média!
        ax.bar(x + deslocamento, invalidos, largura, 
               bottom=df_plot['total_smiles_validos'], color=cor_invalido, edgecolor='white')
        
        # Ajuste dos textos das legendas internas das barras
        for pos_x, altura_y in zip(x + deslocamento, df_plot["total_smiles"]):
            if altura_y == 0:
                continue
            
            # Texto logo acima do topo da barra empilhada
            posicao_texto_y = altura_y + 0.3
            ax.text(pos_x, posicao_texto_y, quant, ha="center", va="bottom", fontsize=12, fontweight='bold', rotation=60)

    ax.grid(axis='y', linestyle='--', alpha=0.2)
    ax.tick_params(axis="y", labelsize=13)
    
    # Ajuste dinâmico do teto do gráfico para a escala de médias (passos menores, ex: de 5 em 5)
    max_altura_com_erro = (df_grouped['total_smiles'] + df_grouped['total_smiles_validos_std']).max()
    y_step = 5
    teto_grafico = int(np.ceil(max_altura_com_erro / float(y_step)) * y_step) + y_step
    
    ax.set_yticks(np.arange(0, teto_grafico, y_step))
    ax.set_ylim(0, teto_grafico)

    ax.set_xlabel('Models', fontsize=15, fontweight='bold', labelpad=10)
    ax.set_ylabel('Mean SMILES per Replica', fontsize=15, fontweight='bold')
    ax.set_title('SMILES Validation per Model and Quantization (Mean ± Std Dev)', fontsize=17, fontweight='bold', pad=15)

    ax.set_xticks(x)
    ax.set_xticklabels(modelos, rotation=15, ha='right', fontsize=14)

    ax.legend(["Unique Valid", "Duplicates", "Invalid"], bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=13)

    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=150, bbox_inches="tight")
    plt.show()

def plotar_grafico_por_replica(df):
    
    # CORREÇÃO 1: Extração correta mantendo o nome intermediário
    df["replica_num"] = df["replica"].apply(lambda x: int(x.split("_replica_")[1]))
    
    # Criamos uma coluna temporária que tira a palavra 'replica' mas mantém o resto (ex: Qwen3.6-35B-BF16)
    df["modelo_quant_temp"] = df["replica"].apply(lambda x: x.split("_replica_")[0])
    
    # Agora sim separamos modelo e quantização do texto completo
    df["modelo"] = df["modelo_quant_temp"].apply(lambda x: x.rsplit("-", 1)[0])
    df["quantizacao"] = df["modelo_quant_temp"].apply(lambda x: x.rsplit("-", 1)[1])

    # Agrupamento
    groupb = df.groupby(["modelo", "quantizacao", "replica_num"])[["total_smiles", "total_smiles_validos", "total_smiles_duplicados"]].mean().reset_index()

    modelo = groupb["modelo"].unique()
    quantizacao = groupb["quantizacao"].unique()

    x = np.arange(len(modelo))
    largura = 0.8 / len(quantizacao)

    # Cores fixas que você escolheu
    cor_valido = "#17ce42"     # Verde
    cor_duplicado = "#4e5de9"  # Azul
    cor_invalido = "#ff0000"   # Vermelho

    # Loop para gerar 5 gráficos separados (um para cada réplica)
    for i in range(1, 6):
        
        # CORREÇÃO 2: Cria uma "folha em branco" nova a cada iteração do loop
        fig, ax = plt.subplots(figsize=(14, 8))
        
        rep = groupb[groupb["replica_num"] ==  i]

        for j, quant in enumerate(quantizacao):
            rep_q = rep[rep["quantizacao"] == quant]

            # Merge para garantir as posições
            df_plot = pd.DataFrame({'modelo': modelo}).merge(rep_q, on='modelo', how='left').fillna(0)

            deslocamento = (j - len(quantizacao)/2) * largura + largura/2

            validos_unicos = df_plot["total_smiles_validos"] - df_plot["total_smiles_duplicados"]
            invalidos = df_plot["total_smiles"] - df_plot["total_smiles_validos"]

            ax.grid(axis='y', linestyle='--', alpha=0.2)  # Adicionar grade horizontal para facilitar a leitura
            ax.set_yticks(np.arange(0, rep['total_smiles'].max() + 20, 20))

            # PLOTAGEM (Lembrete: o que plota primeiro vai pro fundo)
            # 1. Válidos Únicos (Frente)
            ax.bar(x + deslocamento, validos_unicos, largura, color=cor_valido, edgecolor='white', alpha=0.5)

            # 3. Duplicados (Frente) - Assumindo que o número de duplicados é o menor de todos!
            ax.bar(x + deslocamento, df_plot['total_smiles_duplicados'], largura, bottom=validos_unicos, color=cor_duplicado, edgecolor='white')

            # 3. Invalidos (Frente) - O que sobra para completar o total, ou seja, o que ultrapassa os válidos totais
            ax.bar(x + deslocamento, invalidos, largura, bottom=df_plot["total_smiles_validos"],color=cor_invalido, edgecolor='white')

            for pos_x, alt_y in zip(x + deslocamento, df_plot["total_smiles"]):
                if alt_y < 110:
                    ax.text(pos_x, alt_y + 4.5, quant, ha="center", va="bottom", fontsize=12, fontweight='bold', rotation=60)
                else:
                    ax.text(pos_x, (alt_y * 1.7)/2, quant, ha="center", va="top", fontsize=12, fontweight='bold', rotation=60)
            

        # Textos e firulas
        ax.grid(axis='y', linestyle='--', alpha=0.2)
        ax.tick_params(axis="y", labelsize=13)
        ax.set_xlabel('Models', fontsize=15, fontweight='bold', labelpad=10)
        ax.set_ylabel('SMILES Count', fontsize=15, fontweight='bold', labelpad=10)
        # ax.set_ylim(0, df_plot['total_smiles'].max() + 50)
        # O título agora reflete a réplica que estamos olhando
        ax.set_title(f'SMILES Validation per Model and Quantization - Replica {i}', fontsize=17, fontweight='bold', pad=15)
        
        ax.set_xticks(x)
        ax.set_xticklabels(modelo, rotation=45, ha='right', fontsize=13)
        
        ax.legend(["Valid", "Duplicates", "Invalid"], bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=13)
        
        plt.tight_layout()
        plt.savefig(f"grafico_por_replica_{i}.png", dpi=150, bbox_inches="tight")
        plt.show()

def plotar_tempo_resposta_minutos(caminho_json, nome_arquivo_saida):

    # 1. Carregar os dados do arquivo JSON
    with open(caminho_json, "r", encoding="utf-8") as f:
        dados = json.load(f)

    df = pd.DataFrame(dados)

    # 2. Extrair o Modelo e a Quantização a partir da coluna 'model'
    df["modelo"] = df["model"].apply(lambda x: "-".join(x.split("-")[:-1]))
    df["quantizacao"] = df["model"].apply(lambda x: x.split("-")[-1])

    df = df[~df["modelo"].str.contains("Mistral", case=False)]

    # 3. Converter o tempo de resposta para minutos (ms / 60000)
    df["duration_min"] = df["duration_ms"] / 60000.0

    # 4. Remover os modelos Mistral do conjunto de dados
    # df = df[~df["modelo"].str.contains("Mistral", case=False)]
    # print(len(df[df["model"] == "Qwen3.6-35B-BF16"]))

    # 5. Agrupar e calcular a média de 'duration_min'
    df_grouped = (
        df.groupby(["modelo", "quantizacao"])["duration_min"]
        .mean()
        .reset_index()
    )

    print(df_grouped)

    # Definir os modelos filtrados e a ordem fixa das quantizações
    modelos = sorted(df_grouped["modelo"].unique())
    quantizacoes = ["BF16", "Q8", "Q6"]

    x = np.arange(len(modelos))
    largura = 0.8 / len(quantizacoes)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Cores fixas associadas a cada tipo de quantização
    cor_bf16 = "#55a868"  # Verde
    cor_q8 = "#4e5de9"  # Azul
    cor_q6 = "#ff0000"  # Vermelho

    cor_map = {"BF16": cor_bf16, "Q8": cor_q8, "Q6": cor_q6}

    # Limiar dinâmico para decidir a posição do texto na barra
    max_val = df_grouped["duration_min"].max()
    threshold = max_val * 0.75

    for i, quant in enumerate(quantizacoes):
        df_q = df_grouped[df_grouped["quantizacao"] == quant]
        df_plot = (
            pd.DataFrame({"modelo": modelos})
            .merge(df_q, on="modelo", how="left")
            .fillna(0)
        )

        deslocamento = (i - len(quantizacoes) / 2) * largura + largura / 2

        # Plotar as barras correspondentes
        ax.bar(
            x + deslocamento,
            df_plot["duration_min"],
            largura,
            color=cor_map[quant],
            edgecolor="white",
        )

        # Adicionar os rótulos correspondentes a cada barra
        for pos_x, altura_y in zip(x + deslocamento, df_plot["duration_min"]):
            if altura_y == 0:
                continue
            if altura_y < threshold:
                ax.text(
                    pos_x,
                    altura_y + (max_val * 0.01),
                    quant,
                    ha="center",
                    va="bottom",
                    fontsize=13,
                    fontweight="bold",
                    rotation=45,
                )
            else:
                ax.text(
                    pos_x,
                    altura_y / 2,
                    quant,
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                    rotation=45,
                    color="white",
                )

    # Configuração de Grid e Escala em minutos
    ax.grid(axis="y", linestyle="--", alpha=0.2)
    ax.tick_params(axis="y", labelsize=13)

    # Escala dinâmica com passos de 5 em 5 minutos
    y_step = 5
    teto_grafico = np.ceil(max_val / y_step) * y_step + y_step
    ax.set_yticks(np.arange(0, teto_grafico, y_step))
    ax.set_ylim(0, teto_grafico)

    # Títulos e Identificações dos Eixos
    ax.set_xlabel("Models", fontsize=15, fontweight="bold", labelpad=10)
    ax.set_ylabel(
        "Mean Response Time (minutes)", fontsize=15, fontweight="bold"
    )
    ax.set_title(
        "Mean Response Time per Model and Quantization (in Minutes)",
        fontsize=17,
        fontweight="bold",
        pad=15,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(modelos, rotation=15, ha="right", fontsize=14)

    # Legenda customizada
    ax.legend(
        quantizacoes,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        title="Quantization",
        fontsize=13,
        title_fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(nome_arquivo_saida, dpi=150, bbox_inches="tight")
    plt.close()

def plotar_tokens_resposta_minutos(caminho_json, nome_arquivo_saida):

    # 1. Carregar os dados do arquivo JSON
    with open(caminho_json, "r", encoding="utf-8") as f:
        dados = json.load(f)

    df = pd.DataFrame(dados)

    # print(df["tokens"][0]["tokens_per_second"])

    # 2. Extrair o Modelo e a Quantização a partir da coluna 'model'
    df["modelo"] = df["model"].apply(lambda x: "-".join(x.split("-")[:-1]))
    df["quantizacao"] = df["model"].apply(lambda x: x.split("-")[-1])

    # 3. Converter o tempo de resposta para minutos (ms / 60000)
    df["duration_min"] = df["duration_ms"]
    df['token_per_second'] = df['tokens'].apply(lambda x: x['tokens_per_second'])

    df = df[~df["modelo"].str.contains("Mistral", case=False)]

    print(df["token_per_second"])

    # 4. Remover os modelos Mistral do conjunto de dados
    # df = df[~df["modelo"].str.contains("Mistral", case=False)]
    # print(len(df[df["model"] == "Qwen3.6-35B-BF16"]))

    # 5. Agrupar e calcular a média de 'duration_min'
    df_grouped = (
        df.groupby(["modelo", "quantizacao"])["token_per_second"]
        .mean()
        .reset_index()
    )

    print(df_grouped)

    # Definir os modelos filtrados e a ordem fixa das quantizações
    modelos = sorted(df_grouped["modelo"].unique())
    quantizacoes = ["BF16", "Q8", "Q6"]

    x = np.arange(len(modelos))
    largura = 0.8 / len(quantizacoes)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Cores fixas associadas a cada tipo de quantização
    cor_bf16 = "#55a868"  # Verde
    cor_q8 = "#4e5de9"  # Azul
    cor_q6 = "#ff0000"  # Vermelho

    cor_map = {"BF16": cor_bf16, "Q8": cor_q8, "Q6": cor_q6}

    # Limiar dinâmico para decidir a posição do texto na barra
    max_val = df_grouped["token_per_second"].max()
    threshold = max_val * 0.75

    for i, quant in enumerate(quantizacoes):
        df_q = df_grouped[df_grouped["quantizacao"] == quant]
        df_plot = (
            pd.DataFrame({"modelo": modelos})
            .merge(df_q, on="modelo", how="left")
            .fillna(0)
        )

        deslocamento = (i - len(quantizacoes) / 2) * largura + largura / 2

        # Plotar as barras correspondentes
        ax.bar(
            x + deslocamento,
            df_plot["token_per_second"],
            largura,
            color=cor_map[quant],
            edgecolor="white",
        )

        # Adicionar os rótulos correspondentes a cada barra
        for pos_x, altura_y in zip(x + deslocamento, df_plot["token_per_second"]):
            if altura_y == 0:
                continue
            if altura_y < threshold:
                ax.text(
                    pos_x,
                    altura_y + (max_val * 0.01),
                    quant,
                    ha="center",
                    va="bottom",
                    fontsize=13,
                    fontweight="bold",
                    rotation=45,
                )
            else:
                ax.text(
                    pos_x,
                    altura_y / 2,
                    quant,
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                    rotation=45,
                    color="white",
                )

    # Configuração de Grid e Escala em minutos
    ax.grid(axis="y", linestyle="--", alpha=0.2)
    ax.tick_params(axis="y", labelsize=13)

    # Escala dinâmica com passos de 5 em 5 minutos
    y_step = 5
    teto_grafico = np.ceil(max_val / y_step) * y_step + y_step
    ax.set_yticks(np.arange(0, teto_grafico, y_step))
    ax.set_ylim(0, teto_grafico)

    # Títulos e Identificações dos Eixos
    ax.set_xlabel("Models", fontsize=15, fontweight="bold", labelpad=10)
    ax.set_ylabel(
        "Mean Tokens per Second (tokens/s)", fontsize=15, fontweight="bold"
    )
    ax.set_title(
        "Mean Tokens per Second per Model and Quantization",
        fontsize=17,
        fontweight="bold",
        pad=15,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(modelos, rotation=15, ha="right", fontsize=14)

    # Legenda customizada
    ax.legend(
        quantizacoes,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        title="Quantization",
        fontsize=13,
        title_fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(nome_arquivo_saida, dpi=150, bbox_inches="tight")
    plt.close()

if __name__=="__main__":
    # primeira_leva, segunda_leva = preparar_dados(df, per_leva=True)
    task_2 = preparar_dados(df)
    plotar_grafico_media(task_2, 'grafico_task_2_todas_q.png')
    # plotar_grafico_tokens(task_2, 'grafico_tokens_task_2_todas_q.png')
    # dale = plotar_grafico_por_replica(primeira_leva)
    dale = plotar_tokens_resposta_minutos("/home/lucas.abner/Documentos/code/gcrm/teste2_task_2/all_metrics.json", "tokens_segundos_quarta_leva.png")
    dale = plotar_tempo_resposta_minutos("/home/lucas.abner/Documentos/code/gcrm/teste2_task_2/all_metrics.json", "tempo_resposta_min_quarta_leva.png")
    # print(dale)