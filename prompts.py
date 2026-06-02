import requests 
import json
import os
import subprocess

def chamada_llm(message, model_id):

    messages = [
        {
            "role": "user",
            "content": message
        }
    ]

    url = "http://10.30.4.153:8081/v1/chat/completions"

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_id,
        "messages": messages,
        "stream": False,
        "temperature" : 0.10,
        "min_p" : 0.05,
        "top_k" : 0.0,
        "top_p" : 1.0,
        "presence_penalty" : 0.0,
        "frequency_penalty" : 0.0
    }

    response = requests.post(url=url, headers=headers, json=payload)

    with open('metrics.json', 'w', encoding='utf-8') as f:
        json.dump(response.json(), f, indent=4)

    print("Status: ", response.status_code)
    print("Body: ", response.text)

    data = response.json()

    return data["choices"][0]["message"]["content"]

def teste_llms(prompt:str, list_models:list):
    response = requests.get('http://10.30.4.153:8081/api/metrics')
    os.makedirs("./segunda_leva_baixa_temperatura", exist_ok=True)
    for model_id in list_models:
        for i in range(5):
            nome_completo = f"./segunda_leva_baixa_temperatura/{model_id}_replica_{i+1}.txt"
            try:
                response = chamada_llm(prompt, model_id)
                with open(nome_completo, "w", encoding="utf-8") as f:
                    escrita = f.write(response)
                
                # with open('metrics.json', 'w', encoding='utf-8') as f:
                #     json.dump(response.json(), f, indent=4)
                    
            except:
                pass
    return response


if __name__=="__main__":
    # models = ["Qwen3.6-35B-BF16", "gemma-4-26B-Q8", "gemma-4-31B-Q8", "Qwen3.6-35B-Q8", "gemma-4-26B-Q6", "gemma-4-31B-Q6", "Qwen3.6-27B-Q6", "Qwen3.6-35B-Q6", "Ministral3-14B-BF16", "Ministral3-14B-Q8", "Ministral3-14B-Q6", "Magistral1.2-24B-B16", "Magistral1.2-24B-Q8", "Magistral1.2-24B-Q6", "Qwen3.6-27B-BF16",  "Qwen3.6-27B-Q8"]

    # models = ["gemma-4-31B-Q8", "Qwen3.6-35B-Q8", "gemma-4-26B-Q6", "gemma-4-31B-Q6", "Qwen3.6-27B-Q6", "Qwen3.6-35B-Q6", "Ministral3-14B-BF16", "Ministral3-14B-Q8", "Ministral3-14B-Q6", "Magistral1.2-24B-B16", "Magistral1.2-24B-Q8", "Magistral1.2-24B-Q6", "Qwen3.6-27B-BF16",  "Qwen3.6-27B-Q8"]

    models = ["Magistral1.2-24B-B16", "Magistral1.2-24B-Q8", "Magistral1.2-24B-Q6"]

    dale = teste_llms("You are a medicinal chemist. Please generate 100 different molecules. Please make sure to adhere to the following output format restrictions: 1. Report only the SMILES of the molecules in your final answer; 2. Report one SMILES per line and nothing else.", models)

    print(dale)
