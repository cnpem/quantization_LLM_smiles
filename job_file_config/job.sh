#!/bin/bash
#SBATCH --job-name=llama-swap
#SBATCH --output=aichemist_big.out
#SBATCH --error=aichemist_big.err
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=15-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=long-gpu-big

export LLAMA_CACHE='/shared/groups/gcrm/llamacpp_models/'

singularity exec --nv --env LLAMA_CACHE=$LLAMA_CACHE /shared/groups/gcrm/sifs/llama_cpp_swap_v4.sif llama-swap --config /shared/groups/gcrm/config_new.yaml --listen 10.30.4.153:8082 --watch-config

wait
