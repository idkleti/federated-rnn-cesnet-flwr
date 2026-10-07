#!/usr/bin/env bash

# Confronto finale controllato: tutte le configurazioni della figura paper
# usano fraction-train=0.5. Cambiano solo strategia e intervento sui dati.
# Esegui esclusivamente via SLURM:
#   sbatch esperimenti_copernico.sh tools/esperimenti_finali_f50.sh

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

# Riferimento IID controllato: stessi client-size della partizione subnet.
prepara_dati --partition random-sizes
esegui "num-server-rounds=$R fraction-train=0.5 local-epochs=2 lr-decay=0.97 strategy=\"fedavg\" evaluate-central-test=true"

# Quattro strategie sulla partizione reale non-IID; solo FedProx usa mu=1.
prepara_dati
esegui "num-server-rounds=$R fraction-train=0.5 local-epochs=2 lr-decay=0.97 strategy=\"fedavg\" evaluate-central-test=true"
esegui "num-server-rounds=$R fraction-train=0.5 local-epochs=2 lr-decay=0.97 strategy=\"fedprox\" proximal-mu=1.0 evaluate-central-test=true"
esegui "num-server-rounds=$R fraction-train=0.5 local-epochs=1 lr-decay=0.97 strategy=\"fedyogi\" evaluate-central-test=true"
esegui "num-server-rounds=$R fraction-train=0.5 local-epochs=2 lr-decay=0.97 strategy=\"classaware\" evaluate-central-test=true"

# Intervento dati condivisi, a parita' di fraction_train=0.5 con FedAvg.
prepara_dati --mini-dataset "$SHARED_PER_CLASS"
esegui "num-server-rounds=$R fraction-train=0.5 local-epochs=2 lr-decay=0.97 strategy=\"fedavg\" evaluate-central-test=true"

# Ritorna allo scenario standard e rigenera tutte le figure/tabelle.
prepara_dati
python tools/grafici.py
python tools/analizza_tuning_validation.py
