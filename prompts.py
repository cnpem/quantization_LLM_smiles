import requests 
import json
import os
import subprocess


def chamada_llm(message, model_id):
    messages = [{"role": "user", "content": message}]
    url = "http://10.30.4.153:8082/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {"model": model_id, "messages": messages, "stream": False}

    response = requests.post(url=url, headers=headers, json=payload)
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]



def teste_llms(prompt: str, list_models: list):
    leva = "./quarta_leva_menos_moleculas"
    os.makedirs(leva, exist_ok=True)

    for model_id in list_models:
        for i in range(25):
            nome_completo = f"{leva}/{model_id}_replica_{i+1}.txt"
            try:
                content = chamada_llm(prompt, model_id)
                with open(nome_completo, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ {model_id} replica {i+1}")
            except Exception as e:
                print(f"[ERRO] {model_id} replica {i+1}: {e}")

    # Puxa todas as métricas do servidor de uma vez
    metrics_response = requests.get("http://10.30.4.153:8082/api/metrics")
    metrics_response.raise_for_status()
    metrics = metrics_response.json()

    with open(f"{leva}/all_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"Métricas salvas: {len(metrics)} entradas")
    return metrics



if __name__=="__main__":
    # models = ["Qwen3.6-35B-BF16", "gemma-4-26B-Q8", "gemma-4-31B-Q8", "Qwen3.6-35B-Q8", "gemma-4-26B-Q6", "gemma-4-31B-Q6", "Qwen3.6-27B-Q6", "Qwen3.6-35B-Q6", "Ministral3-14B-BF16", "Ministral3-14B-Q8", "Ministral3-14B-Q6", "Magistral1.2-24B-B16", "Magistral1.2-24B-Q8", "Magistral1.2-24B-Q6", "Qwen3.6-27B-BF16",  "Qwen3.6-27B-Q8"]

    # models = ["gemma-4-31B-Q8", "Qwen3.6-35B-Q8", "gemma-4-26B-Q6", "gemma-4-31B-Q6", "Qwen3.6-27B-Q6", "Qwen3.6-35B-Q6", "Ministral3-14B-BF16", "Ministral3-14B-Q8", "Ministral3-14B-Q6", "Magistral1.2-24B-B16", "Magistral1.2-24B-Q8", "Magistral1.2-24B-Q6", "Qwen3.6-27B-BF16",  "Qwen3.6-27B-Q8"]

    models = [
        "gpt-oss-120B-BF16",
        "gpt-oss-120B-Q8",
        "gpt-oss-120B-Q6",
        "gemma-4-31B-BF16",
        "gemma-4-31B-Q8",
        "gemma-4-31B-Q6",
        "Qwen3.6-35B-BF16",
        "Qwen3.6-35B-Q8",
        "Qwen3.6-35B-Q6",
        "Mistral3.2-24B-BF16",
        "Mistral3.2-24B-Q8",
        "Mistral3.2-24B-Q6",
    ]

    dale = teste_llms("You are a medicinal chemist. Please generate 20 different molecules. Please make sure to adhere to the following output format restrictions: 1. Report only the SMILES of the molecules in your final answer; 2. Report one SMILES per line and nothing else.", models)

    print(dale)
