#!/usr/bin/env bash

# FedAvgM e FedAdagrad sulla stessa partizione non-IID (subnet) e pipeline
# delle altre baseline. Da eseguire esclusivamente tramite il job SLURM:
#   sbatch esperimenti_copernico.sh tools/esperimenti_serveropt.sh

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd -- "$script_dir/.."

# Override utile per un controllo breve: R=1 N=1 sbatch ...
R="${R:-25}"
N="${N:-5}"
FEDERATION_CONFIG="${FEDERATION_CONFIG:-num-supernodes=69 client-resources-num-cpus=11 client-resources-num-gpus=0.5}"

prepare_data_args=()
if [[ "${FEDRNN_OFFLINE:-0}" == "1" ]]; then
    prepare_data_args=(--offline)
fi

prepara_dati() {
    python prepare_data.py "${prepare_data_args[@]}" "$@"
}

esegui() {
    echo "==> $*"
    for i in $(seq 1 "$N"); do
        printf '  [%2d/%2d] %s  ' "$i" "$N" "$(date +%H:%M:%S)"
        if ! output=$(flwr run . \
            --federation-config "$FEDERATION_CONFIG" \
            --run-config "$*" --stream 2>&1); then
            printf '%s\n' "$output" >&2
            return 1
        fi
        printf '%s\n' "$output"
        printf '%s\n' "$output" | grep -oE "checkpoint selezionato: [0-9.]+" | tail -1 || true
    done
}

# Mantiene dati, client, fraction_train, LR locale e decadimento delle baseline
# principali; cambiano solo l'ottimizzatore lato server e i suoi parametri.
prepara_dati
esegui "num-server-rounds=$R fraction-train=0.5 lr-decay=0.97 strategy=\"fedavgm\" server-learning-rate=1.0 server-momentum=0.9"
esegui "num-server-rounds=$R fraction-train=0.5 lr-decay=0.97 strategy=\"fedadagrad\" server-learning-rate=0.1 server-tau=0.001"
