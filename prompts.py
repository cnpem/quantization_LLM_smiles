import requests 
import json

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
        "stream": False
    }

    response = requests.post(url=url, headers=headers, json=payload)

    print("Status: ", response.status_code)
    print("Body: ", response.text)

    data = response.json()

    return data["choices"][0]["message"]["content"]

def teste_llms(prompt:str, list_models:list):
    for model_id in list_models:
        for i in range(2):
            nome_completo = f"{model_id}_replica_{i+1}.txt"
            response = chamada_llm(prompt, model_id)
            with open(nome_completo, "w", encoding="utf-8") as f:
                escrita = f.write(response)
    return response


if __name__=="__main__":
    # models = ["gemma-4-26B-BF16", "gemma-4-31B-BF16", "Mistral3.2-24B-BF16", "Qwen3.6-27B-BF16", "Qwen3.6-35B-BF16", "gemma-4-26B-Q8", "gemma-4-31B-Q8", "Mistral3.2-24B-Q8", "Qwen3.6-27B-Q8", "Qwen3.6-35B-Q8", "gemma-4-26B-Q6", "gemma-4-31B-Q6", "Mistral3.2-24B-Q6", "Qwen3.6-27B-Q6", "Qwen3.6-35B-Q6"]

    models = ["Qwen3.6-35B-Q6"]

    # models = ["gpt-oss-20B-Q6"]
    dale = teste_llms("You are a medicinal chemist. Please generate 100 different molecules. Please make sure to adhere to the following output format restrictions: 1. Report only the SMILES of the molecules in your final answer; 2. Report one SMILES per line and nothing else.", models)

    print(dale)
