#!/usr/bin/env bash

# Valutazione finale delle strategie selezionate senza guardare il test durante
# il tuning. I parametri sotto provengono da
# outputs/tables/validation_tuning_summary.csv (run validation-only del 2026-10-01).
# Il checkpoint continua a essere scelto dalla validation federata; il test
# centrale viene valutato una sola volta, solo per la misura finale.
#
# Esegui esclusivamente via SLURM:
#   sbatch esperimenti_copernico.sh tools/esperimenti_finali.sh

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd -- "$script_dir/.."

# Stesso orizzonte del tuning (25 round) e cinque repliche, così il confronto
# finale resta direttamente comparabile alla selezione via validation.
# Per un controllo breve: R=1 N=1 sbatch ...
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
        printf '%s\n' "$output" | grep -oE 'macro-F1 sul test, checkpoint selezionato: [0-9.]+' | tail -1 || true
    done
}

# Mantiene la partizione naturale subnet e lr-decay=0.97 del tuning. Tutte le
# opzioni sono esplicite per rendere la tabella di configurazione riproducibile.
prepara_dati

# Migliori configurazioni per strategy, ordinate nella validation-only summary:
esegui "num-server-rounds=$R fraction-train=1.0 local-epochs=2 lr-decay=0.97 strategy=\"fedavg\" evaluate-central-test=true"
esegui "num-server-rounds=$R fraction-train=0.5 local-epochs=2 lr-decay=0.97 strategy=\"fedprox\" proximal-mu=1.0 evaluate-central-test=true"
esegui "num-server-rounds=$R fraction-train=1.0 local-epochs=1 lr-decay=0.97 strategy=\"fedyogi\" evaluate-central-test=true"
esegui "num-server-rounds=$R fraction-train=0.5 local-epochs=2 lr-decay=0.97 strategy=\"classaware\" evaluate-central-test=true"

# Aggiorna le figure/tabelle standard e figure_tables.zip. Grafici.py considera
# soltanto gli storici con un test finale valido, quindi esclude il tuning.
python tools/grafici.py
