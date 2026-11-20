#!/usr/bin/env bash
#SBATCH --job-name=fl_epochs
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=22
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --qos=gpuResA_qos
#SBATCH --partition=gpu_H100

set -euo pipefail

# Riusa l'attivazione Conda/CUDA e i controlli offline del job principale.
cd /hpc/home/clmlnz/federated-rnn-cesnet-flwr
exec bash esperimenti_copernico.sh tools/esperimenti_ablazione_epoche.sh
