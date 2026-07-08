import requests 
import json
import os
import argparse
import config

def chamada_llm(message, model_id):
    messages = [{"role": "user", "content": message}]
    headers = {"Content-Type": "application/json"}
    payload = {"model": model_id, "messages": messages, "stream": False}

    response = requests.post(url=config.API_URL, headers=headers, json=payload)
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]

def teste_llms(task_name: str, prompt: str, list_models: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    arquivo_metricas = os.path.join(output_dir, "all_metrics.json")

    for model_id in list_models:
        for i in range(config.N_REPLICAS):
            nome_completo = os.path.join(output_dir, f"{model_id}_replica_{i+1}.txt")
            
            try:
                content = chamada_llm(prompt, model_id)
                with open(nome_completo, "w", encoding="utf-8") as f:
                    f.write(content)
                
                print(f"✅ [{task_name}] {model_id} replica {i+1}")
                
                metrics_response = requests.get(config.API_METRICS_URL)
                metrics_response.raise_for_status()
                metrics_model = metrics_response.json()

                with open(arquivo_metricas, "w", encoding="utf-8") as f:
                    json.dump(metrics_model, f, indent=4)

                print(f"   Métricas salvas.")
                
            except Exception as e:
                print(f"❌ [ERRO] {model_id} replica {i+1}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coletar dados das LLMs (Geração de txts)")
    parser.add_argument("--task", type=str, help="Executar apenas uma task especifica (ex: task_1)")
    args = parser.parse_args()

    print("==================================================")
    print("  GCRM - Coleta de Dados via API LLM")
    print("==================================================")
    print(f"Réplicas configuradas: {config.N_REPLICAS}")
    print(f"Modelos configurados:  {len(config.MODELS)}")
    print(f"API configurada:       {config.API_URL}")
    print("==================================================\n")

    for t_name, t_config in config.TASKS.items():
        if args.task and t_name != args.task:
            continue
            
        print(f"Iniciando coleta para: {t_name}...")
        teste_llms(
            task_name=t_name,
            prompt=t_config["prompt"],
            list_models=config.MODELS,
            output_dir=str(t_config["raw_dir"])
        )
        print(f"--- Coleta de {t_name} finalizada! ---\n")
