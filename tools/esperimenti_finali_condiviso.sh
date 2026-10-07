#!/usr/bin/env bash

# Valutazione finale della baseline con piccolo dataset condiviso: 500 serie
# per classe sono distribuite a tutti i client, mentre la partizione nativa
# subnet resta non-IID. Usa FedAvg con gli iperparametri selezionati dal tuning
# validation-only, cosi' il confronto con FedAvg non-IID e' diretto.
#
# Esegui esclusivamente via SLURM:
#   sbatch esperimenti_copernico.sh tools/esperimenti_finali_condiviso.sh

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd -- "$script_dir/.."

R="${R:-25}"
N="${N:-5}"
SHARED_PER_CLASS="${SHARED_PER_CLASS:-500}"
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
        printf '%s\n' "$output" | grep -oE 'macro-F1 sul test, checkpoint selezionato: [0-9.]+' | tail -1 || true
    done
}

# La partizione resta subnet; solo le serie condivise cambiano il training.
prepara_dati --mini-dataset "$SHARED_PER_CLASS"
esegui "num-server-rounds=$R fraction-train=1.0 local-epochs=2 lr-decay=0.97 strategy=\"fedavg\" evaluate-central-test=true"

# Riporta gli shard allo scenario non-IID standard: grafici.py legge meta.json
# anche per la figura descrittiva della partizione.
prepara_dati
python tools/grafici.py
python tools/analizza_tuning_validation.py
