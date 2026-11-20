#!/usr/bin/env bash
#SBATCH --job-name=fl_split451045
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=22
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --qos=gpuResA_qos
#SBATCH --partition=gpu_L40S

set -euo pipefail

cd /hpc/home/clmlnz/federated-rnn-cesnet-flwr
exec bash esperimenti_copernico.sh tools/esperimenti_finali_split_45_10_45.sh
