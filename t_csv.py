import numpy as np 
import pandas as pd
from pathlib import Path
import os

# arquivo_txt = "gemma-4-26B-BF16_replica_1.txt"
# arquivo_csv = f"{Path(__file__).parent}/gemma-4-26B-BF16_replica_1.csv"
pasta_txt = Path(__file__).parent/"segunda_leva_baixa_temperatura"
pasta_csv = Path(__file__).parent/"csv_llms_segunda_leva"

pasta_csv.mkdir(exist_ok=True)

for arquivo in pasta_txt.glob("*.txt"):
    with open(arquivo, "r", encoding="utf-8") as f:
        conteudo = f.read()
        conteudo_lista = [linha for linha in conteudo.split("\n") if linha.strip()]

    df = pd.DataFrame(conteudo_lista, columns=["SMILES"])

    arquivo_csv = arquivo.with_suffix(".csv").name
    
    caminho_salvamento = pasta_csv / arquivo_csv

    df.to_csv(caminho_salvamento, index=False)

if __name__=="__main__":
    print("DALE")