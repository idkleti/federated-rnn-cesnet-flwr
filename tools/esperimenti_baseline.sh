#!/usr/bin/env bash

# Baseline aggiuntive sulla stessa pipeline di esperimenti.sh.
# Non scarica dati: con FEDRNN_OFFLINE=1 rigenera gli shard esclusivamente dai
# dati gia' presenti, come il job Copernico principale.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd -- "$script_dir/.."

# Per il debug il chiamante puo' ridurre le run senza modificare lo script:
#   R=1 N=1 bash tools/esperimenti_baseline.sh
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

# Stessa partizione subnet e stessi iperparametri della baseline principale:
# il confronto isola esclusivamente l'algoritmo di aggregazione.
prepara_dati
# FedNova e' definita qui per SGD senza momentum; le altre baseline mantengono
# l'Adam configurato in pyproject.toml.
esegui "num-server-rounds=$R fraction-train=0.5 lr-decay=0.97 strategy=\"fednova\" local-optimizer=\"sgd\""
esegui "num-server-rounds=$R fraction-train=0.5 lr-decay=0.97 strategy=\"classaware\""
