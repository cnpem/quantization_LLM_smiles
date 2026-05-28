import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Carregar os dados
df = pd.read_csv('results_smiles/results.csv')

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

        primeira_leva = df_c[df_c['leva'] == 'primeira_leva']
        segunda_leva = df_c[df_c['leva'] == 'segunda_leva']
        return primeira_leva, segunda_leva
    else:
        return df

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plotar_grafico(leva, nome_arquivo):
    df_grouped = leva.groupby(['modelo', 'quantizacao'])[['total_smiles', 'total_smiles_validos', 'total_smiles_duplicados']].sum().reset_index()

    modelos = df_grouped['modelo'].unique()
    quantizacoes = df_grouped['quantizacao'].unique()

    x = np.arange(len(modelos))
    largura = 0.8 / len(quantizacoes)

    fig, ax = plt.subplots(figsize=(16, 8))

    # Cores fixas que você escolheu
    cor_valido = "#55a868"     # Verde
    cor_duplicado = "#4e5de9"  # Azul
    cor_invalido = "#ff0000"   # Vermelho

    for i, quant in enumerate(quantizacoes):
        df_q = df_grouped[df_grouped['quantizacao'] == quant]
        df_plot = pd.DataFrame({'modelo': modelos}).merge(df_q, on='modelo', how='left').fillna(0)
        
        deslocamento = (i - len(quantizacoes)/2) * largura + largura/2
        
        # MATEMÁTICA DO EMPILHAMENTO
        # Para empilhar sem ultrapassar o "Total de SMILES", precisamos das fatias exatas:
        validos_unicos = df_plot['total_smiles_validos'] - df_plot['total_smiles_duplicados']
        invalidos = df_plot['total_smiles'] - df_plot['total_smiles_validos']
        
        # PLOT 1: VÁLIDOS ÚNICOS (Começa no 0, fica na BASE)
        ax.bar(x + deslocamento, validos_unicos, largura, color=cor_valido, edgecolor='white')

        # PLOT 2: DUPLICADOS (Começa onde os válidos únicos terminam, fica no MEIO)
        ax.bar(x + deslocamento, df_plot['total_smiles_duplicados'], largura, 
            bottom=validos_unicos, color=cor_duplicado, edgecolor='white')

        # PLOT 3: INVÁLIDOS (Começa onde os válidos totais terminam, fica no TOPO)
        ax.bar(x + deslocamento, invalidos, largura, 
            bottom=df_plot['total_smiles_validos'], color=cor_invalido, edgecolor='white')
        
        for pos_x, altura_y in zip(x + deslocamento, df_plot["total_smiles"]):
            if altura_y < 550:
                ax.text(pos_x, altura_y + 4.5, quant, ha="center", va="bottom", fontsize=9, fontweight='bold', rotation=60)
            else:
                ax.text(pos_x, (altura_y * 0.9)/2, quant, ha="center", va="top", fontsize=9, fontweight='bold', rotation=60)

    # Grid e Escala (movido para fora do loop para não repetir à toa)
    ax.grid(axis='y', linestyle='--', alpha=0.2)
    teto_grafico = df_grouped['total_smiles'].max() + 50
    ax.set_yticks(np.arange(0, teto_grafico, 50))

    ax.set_ylim(0, 600)  # Ajuste o limite superior conforme necessário

    ax.set_xlabel('Modelos', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Quantidade de SMILES', fontsize=12, fontweight='bold')
    ax.set_title('Validação de SMILES por Modelo e Quantização', fontsize=14, fontweight='bold', pad=15)

    ax.set_xticks(x)
    ax.set_xticklabels(modelos, rotation=45, ha='right', fontsize=10)

    # Legenda ajustada
    ax.legend(["Válidos", "Duplicados", "Inválidos"], bbox_to_anchor=(1.01, 1), loc='upper left')

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
    groupb = df.groupby(["modelo", "quantizacao", "replica_num"])[["total_smiles", "total_smiles_validos", "total_smiles_duplicados"]].sum().reset_index()

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
                    ax.text(pos_x, alt_y + 4.5, quant, ha="center", va="bottom", fontsize=9, fontweight='bold', rotation=60)
                else:
                    ax.text(pos_x, (alt_y * 1.7)/2, quant, ha="center", va="top", fontsize=9, fontweight='bold', rotation=60)
            

        # Textos e firulas
        ax.set_xlabel('Modelos', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_ylabel('Quantidade de SMILES', fontsize=12, fontweight='bold', labelpad=10)
        # ax.set_ylim(0, df_plot['total_smiles'].max() + 50)
        # O título agora reflete a réplica que estamos olhando
        ax.set_title(f'Validação de SMILES por Modelo e Quantização - Réplica {i}', fontsize=14, fontweight='bold', pad=15)
        
        ax.set_xticks(x)
        ax.set_xticklabels(modelo, rotation=45, ha='right', fontsize=10)
        
        ax.legend(["Válidos", "Duplicados", "Inválidos"], bbox_to_anchor=(1.01, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"grafico_por_replica_{i}.png", dpi=150, bbox_inches="tight")
        plt.show()

if __name__=="__main__":
    primeira_leva, segunda_leva = preparar_dados(df, per_leva=True)
    plotar_grafico(primeira_leva, 'grafico_primeira_leva.png')
    dale = plotar_grafico_por_replica(primeira_leva)
    print(dale)