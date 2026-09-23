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

# Imposta la directory di output
export FEDRNN_OUTPUT_ROOT="$PROJECT_ROOT/outputs"
mkdir -p "$FEDRNN_OUTPUT_ROOT"
#ne faccio anche sopra per sicurezza
mkdir -p /hpc/home/clmlnz/outputs
echo "FEDRNN_OUTPUT_ROOT=$FEDRNN_OUTPUT_ROOT"
echo "Output directory: $(readlink -f "$FEDRNN_OUTPUT_ROOT")"

# Ogni job usa uno stato Flower separato fuori dal repository: evita conflitti
# SQLite senza far includere le app generate nelle scansioni successive.
export FLWR_HOME="$HOME/.flwr-${SLURM_JOB_ID:-manual}"
export FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1
mkdir -p "$FLWR_HOME"
echo "FLWR_HOME=$FLWR_HOME"


# Il dataset è stato trasferito in precedenza da una macchina con Internet.
# FEDRNN_OFFLINE fa sì che ogni rigenerazione degli shard in esperimenti.sh
# rifiuti download impliciti.
export CESNET_DATA_ROOT="$PROJECT_ROOT/time_dataset"
export FEDRNN_OFFLINE=1
export PYTHONUNBUFFERED=1
# Il nodo di calcolo non ha accesso a PyPI: usa le dipendenze gia' presenti
# nell'ambiente Conda invece di creare un runtime Flower con uv sync.
export FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1

if [[ ! -f "$CESNET_DATA_ROOT/.fedrnn-cesnet-ready.json" ]]; then
    echo "Dataset offline non pronto: manca $CESNET_DATA_ROOT/.fedrnn-cesnet-ready.json" >&2
    echo "Esegui prima python prepare_data.py --download-only sul nodo di login." >&2
    exit 1
fi

echo "Current working directory: $(pwd) - Running on $(hostname)"

# Per retrocompatibilita' senza argomenti esegue la suite storica. Per le nuove
# baseline: sbatch esperimenti_copernico.sh tools/esperimenti_baseline.sh
# server-side optimizer: sbatch esperimenti_copernico.sh tools/esperimenti_serveropt.sh
EXPERIMENT_SCRIPT="${1:-tools/esperimenti.sh}"
case "$EXPERIMENT_SCRIPT" in
    tools/esperimenti.sh|tools/esperimenti_baseline.sh|tools/esperimenti_serveropt.sh) ;;
    *)
        echo "Script esperimenti non consentito: $EXPERIMENT_SCRIPT" >&2
        echo "Valori ammessi: tools/esperimenti.sh, tools/esperimenti_baseline.sh, tools/esperimenti_serveropt.sh" >&2
        exit 2
        ;;
esac
echo "Suite esperimenti: $EXPERIMENT_SCRIPT"
bash "$EXPERIMENT_SCRIPT"

echo "Job finished on $(date)"
