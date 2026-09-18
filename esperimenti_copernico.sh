#!/bin/bash
#SBATCH --job-name=fl_CESNET
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=22
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --qos=gpuResA_qos
#SBATCH --partition=gpu_H100

set -euo pipefail

echo "Job started on $(date)"

# Attiva ambiente virtuale
#source /hpc/home/clmlnz/tesi/309/bin/activate
module load miniconda3/24.4.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate 312_l40s

# Carica modulo CUDA
module load cuda/12.2

# Imposta la directory di lavoro
PROJECT_ROOT=/hpc/home/clmlnz/federated-rnn-cesnet-flwr
cd "$PROJECT_ROOT"

# Il dataset è stato trasferito in precedenza da una macchina con Internet.
# FEDRNN_OFFLINE fa sì che ogni rigenerazione degli shard in esperimenti.sh
# rifiuti download impliciti.
export CESNET_DATA_ROOT="$PROJECT_ROOT/time_dataset"
export FEDRNN_OFFLINE=1
export PYTHONUNBUFFERED=1

if [[ ! -f "$CESNET_DATA_ROOT/.fedrnn-cesnet-ready.json" ]]; then
    echo "Dataset offline non pronto: manca $CESNET_DATA_ROOT/.fedrnn-cesnet-ready.json" >&2
    echo "Esegui prima python prepare_data.py --download-only sul nodo di login." >&2
    exit 1
fi

echo "Current working directory: $(pwd) - Running on $(hostname)"

# Esegui lo script Python
bash tools/esperimenti.sh

echo "Job finished on $(date)"