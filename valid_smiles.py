import numpy as np
import pandas as pd
from rdkit import Chem 
from rdkit.Chem import Descriptors, AllChem
from rdkit import RDLogger
from pathlib import Path
import os
import json

RDLogger.DisableLog("rdApp.*")

def validate_smiles(list_smiles):

    lst = []

    for smiles in list_smiles:
        try:
            canonical_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
            lst.append(canonical_smiles)
        except:
            pass
    
    return lst

def split_smiles(csv_smiles, nome_leva):

    df_smiles = pd.read_csv(csv_smiles, sep=",")

    lst_smiles = df_smiles["SMILES"].tolist()
    print("Total de Smiles gerados nesse ciclo: ", len(lst_smiles))

    lst_valid_smiles = validate_smiles(lst_smiles)
    print("Total de Smiles validos nesse ciclo: ", len(lst_valid_smiles))

    # smiles_duplicados = [x for x in lst_valid_smiles if lst_valid_smiles.count(x) > 1]
    smiles_duplicados = df_smiles.duplicated(subset=["SMILES"]).sum()
    print("Total de Smiles duplicados nesse ciclo: ", smiles_duplicados)
        

    arquivo_split = csv_smiles.split("/")

    novo_df = pd.DataFrame({"task": [task_name], "replica": [arquivo_split[2].replace(".csv", "")], "total_smiles":[len(lst_smiles)], "total_smiles_validos": [len(lst_valid_smiles)], "total_smiles_duplicados": [smiles_duplicados]})

    caminho_results = "./results_smiles"
    caminho_arquivo_csv = f"{caminho_results}/results_{nome_leva}_agregado.csv"
    arquivo_existe = os.path.isfile(caminho_arquivo_csv)

    # mode='a' faz o append, adicionando a linha no final
    novo_df.to_csv(caminho_arquivo_csv, mode='a', index=False, header=not arquivo_existe)

    return "Deu certo"

def split_total(csv_smiles="", metricas="", task_name=""):
    # 1. Carregar o log completo de métricas (JSON)
    with open(metricas, "r", encoding="utf-8") as f:
        df_metricas = json.load(f)
    df_metricas = pd.DataFrame(df_metricas)

    # 2. Descobrir qual modelo e quantização estamos processando neste ciclo
    # Exemplo: 'gemma-4-31B-Q8_replica_8.csv' -> 'gemma-4-31B-Q8'
    arquivo_nome = csv_smiles.split("/")[-1]
    modelo_com_quant = arquivo_nome.split("_replica")[0]

    # 3. FILTRAR O JSON: Pegar apenas as linhas do modelo correto desta réplica
    df_modelo_atual = df_metricas[df_metricas["model"] == modelo_com_quant].copy()

    # Se por acaso o modelo não for encontrado no JSON, evita quebrar o código
    if df_modelo_atual.empty:
        print(
            f"Aviso: O modelo {modelo_com_quant} não foi encontrado no arquivo de métricas."
        )
        return "Modelo não encontrado"

    # 4. Ler e processar o CSV de SMILES
    df_smiles = pd.read_csv(csv_smiles, sep=",")
    lst_smiles = df_smiles["SMILES"].tolist()
    lst_valid_smiles = validate_smiles(lst_smiles)
    smiles_duplicados = df_smiles.duplicated(subset=["SMILES"]).sum()

    print(f"\n--- Processando: {modelo_com_quant} ---")
    print("Total de Smiles gerados: ", len(lst_smiles))
    print("Total de Smiles válidos: ", len(lst_valid_smiles))
    print("Total de Smiles duplicados: ", smiles_duplicados)

    # 5. Calcular as métricas de tempo e velocidade APENAS para este modelo
    df_modelo_atual["duration_min"] = df_modelo_atual["duration_ms"] / 60000.0
    df_modelo_atual["token_per_second"] = df_modelo_atual["tokens"].apply(
        lambda x: x["tokens_per_second"]
    )

    # 6. Separar o Modelo Base da Quantização para salvar estruturado
    modelo_base = "-".join(modelo_com_quant.split("-")[:-1])  # Ex: gemma-4-31B
    quantizacao = modelo_com_quant.split("-")[-1]  # Ex: Q8

    # 7. Criar a linha única consolidada deste modelo/quantização
    total = len(lst_smiles)
    validos = len(lst_valid_smiles)
    invalidos = total - validos

    pct_validos = (validos / total) * 100 if total > 0 else 0.0
    pct_invalidos = (invalidos / total) * 100 if total > 0 else 0.0

    novo_df = pd.DataFrame(
        {
            "task": [task_name],
            "modelo": [modelo_base],
            "quantizacao": [quantizacao],
            "replica": [arquivo_nome.replace(".csv", "")],
            "total_smiles": [total],
            "total_smiles_validos": [validos],
            "total_smiles_duplicados": [smiles_duplicados],
            "pct_validos": [pct_validos],
            "pct_invalidos": [pct_invalidos],
            "duration_min_media": [
                df_modelo_atual["duration_min"].mean()
            ],  # Média do modelo correto
            "token_per_second_medio": [
                df_modelo_atual["token_per_second"].mean()
            ],  # Média do modelo correto
        }
    )

    df_grouped = novo_df.groupby(['modelo', 'quantizacao'])[['total_smiles', 'total_smiles_validos', 'total_smiles_duplicados']].sum().reset_index()
    print(df_grouped)


    # 8. Salvar fazendo Append (adicionando como nova linha)
    caminho_results = "./results_smiles"
    os.makedirs(caminho_results, exist_ok=True)  # Garante que a pasta existe
    caminho_arquivo_csv = f"{caminho_results}/results_{task_name}.csv"
    arquivo_existe = os.path.isfile(caminho_arquivo_csv)

    novo_df.to_csv(
        caminho_arquivo_csv,
        mode="a",
        index=False,
        header=not arquivo_existe,
    )

    return "Deu certo"

if __name__=="__main__":
    
    for pasta in os.listdir("./"):
        caminho_pasta = os.path.join("./", pasta)
        
        if os.path.isdir(caminho_pasta) and pasta.startswith("csv_llms_task_2"):

            nome_leva = pasta.replace("csv_llms_", "")

            for csv in os.listdir(caminho_pasta):
                caminho_final = os.path.join(caminho_pasta, csv)
                print(caminho_final)
                dale = split_total(csv_smiles=caminho_final, metricas="/home/lucas.abner/Documentos/code/gcrm/teste2_task_2/all_metrics.json",task_name="task_2")
                print(dale)
    df = pd.read_csv('/home/lucas.abner/Documentos/code/gcrm/results_smiles/results_task_2.csv')
    df_grouped = df.groupby(['modelo', 'quantizacao'])[['total_smiles', 'total_smiles_validos', 'total_smiles_duplicados']].sum().reset_index()
    df_group_duration = df.groupby(["modelo", "quantizacao"])[["token_per_second_medio", "duration_min_media"]].mean().reset_index()

    df_total = pd.merge(
        df_grouped, df_group_duration, on=["modelo", "quantizacao"]
    )

    df_total["% smiles validos"] = (df_total["total_smiles_validos"] / df_total["total_smiles"]) * 100

    print(df_total)

    caminho_results = "./results_smiles"
    os.makedirs(caminho_results, exist_ok=True)  # Garante que a pasta existe
    caminho_arquivo_csv = f"{caminho_results}/results_task_2_agregado.csv"
    arquivo_existe = os.path.isfile(caminho_arquivo_csv)

    df_total.to_csv(
        caminho_arquivo_csv,
        mode="a",
        index=False,
        header=not arquivo_existe,
    )