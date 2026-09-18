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

set -e

echo "Job started on $(date)"

# Attiva ambiente virtuale
#source /hpc/home/clmlnz/tesi/309/bin/activate
module load miniconda3/24.4.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate 312_l40s

# Carica modulo CUDA
module load cuda/12.2
LD_LIBRARY_PATH="$LD_LIBRARY_PATH"64

# Imposta la directory di lavoro
cd /hpc/home/clmlnz/federated-rnn-cesnet-flwr

echo "Current working directory: $(pwd) - Running on $(hostname)"

# Esegui lo script Python
bash tools/esperimenti.sh

echo "Job finished on $(date)"