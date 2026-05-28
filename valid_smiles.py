import numpy as np
import pandas as pd
from rdkit import Chem 
from rdkit.Chem import Descriptors, AllChem
from rdkit import RDLogger
from pathlib import Path
import os

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

    # for smi in lst_smiles:
    #     # Pega o SMILES canônico para fazer a comparação correta
    #     mol = Chem.MolFromSmiles(str(smi))
    #     if mol is not None:
    #         print(f"O SMILE {smi} é válido!")
    #     else: 
    #         print(f"O SMILE {smi} é invalido!")

        

    arquivo_split = csv_smiles.split("/")

    novo_df = pd.DataFrame({"leva": [nome_leva], "replica": [arquivo_split[2].replace(".csv", "")], "total_smiles":[len(lst_smiles)], "total_smiles_validos": [len(lst_valid_smiles)], "total_smiles_duplicados": [smiles_duplicados]})

    caminho_results = "./results_smiles"
    caminho_arquivo_csv = f"{caminho_results}/results.csv"
    arquivo_existe = os.path.isfile(caminho_arquivo_csv)

    # mode='a' faz o append, adicionando a linha no final
    novo_df.to_csv(caminho_arquivo_csv, mode='a', index=False, header=not arquivo_existe)

    return "Deu certo"


if __name__=="__main__":
    
    for pasta in os.listdir("./"):
        caminho_pasta = os.path.join("./", pasta)
        
        if os.path.isdir(caminho_pasta) and pasta.startswith("csv_"):

            nome_leva = pasta.replace("csv_llms_", "")

            for csv in os.listdir(caminho_pasta):
                caminho_final = os.path.join(caminho_pasta, csv)
                print(caminho_final)
                dale = split_smiles(caminho_final, nome_leva)
                print(dale)