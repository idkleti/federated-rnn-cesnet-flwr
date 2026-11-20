#!/usr/bin/env bash

# Ripete la suite finale f=0.5 con una ripartizione effettiva:
#   45% train federato, 10% validation federata, 45% test centrale.
# Nessun indirizzo viene scartato: il 55% non-test viene diviso localmente
# nei client, trattenendo il 18.1818% per la validation (10% totale).
# Le run sono isolate da quelle standard in outputs/split_45_10_45.
#
# Avvio: sbatch esperimenti_split_45_10_45.sbatch

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd -- "$script_dir/.."

export FEDRNN_TEST_RATIO=0.45
export FEDRNN_HOLDOUT_VAL_RATIO=0
export FEDRNN_CLIENT_VAL_FRACTION=0.18181818181818182
export FEDRNN_OUTPUT_ROOT="$PWD/outputs/split_45_10_45"
mkdir -p "$FEDRNN_OUTPUT_ROOT"

R="${R:-25}"
N="${N:-5}"
SHARED_PER_CLASS="${SHARED_PER_CLASS:-500}"
RUN_ADDITIONAL_ONLY="${RUN_ADDITIONAL_ONLY:-0}"
FEDERATION_CONFIG="${FEDERATION_CONFIG:-client-resources-num-cpus=11 client-resources-num-gpus=0.5}"

prepara_dati() {
    python prepare_data.py --offline "$@"

    # Con il nuovo split alcune subnet possono scendere sotto la soglia minima:
    # il numero di SuperNode deve quindi seguire gli shard appena generati.
    local num_supernodes
    num_supernodes="$(python -c 'import json; print(json.load(open("shards/meta.json", encoding="utf-8"))["num_partitions"])')"
    if [[ "$FEDERATION_CONFIG" =~ (^|[[:space:]])num-supernodes= ]]; then
        FEDERATION_CONFIG="$(printf '%s\n' "$FEDERATION_CONFIG" | sed -E "s/(^|[[:space:]])num-supernodes=[^[:space:]]+/\\1num-supernodes=$num_supernodes/")"
    else
        FEDERATION_CONFIG="num-supernodes=$num_supernodes $FEDERATION_CONFIG"
    fi
    echo "Federation config: $FEDERATION_CONFIG"
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

esegui_o_fallisci() {
    if ! esegui "$@"; then
        echo "Run fallita; interrompo la suite." >&2
        exit 1
    fi
}

# Stesse configurazioni di tools/esperimenti_finali_f50.sh; cambiano soltanto
# lo split e la cartella di output.
if [[ "$RUN_ADDITIONAL_ONLY" != "1" ]]; then
    prepara_dati --partition random-sizes
    esegui_o_fallisci "num-server-rounds=$R fraction-train=0.5 local-epochs=2 lr-decay=0.97 strategy=\"fedavg\" evaluate-central-test=true"
fi

prepara_dati
if [[ "$RUN_ADDITIONAL_ONLY" != "1" ]]; then
    esegui_o_fallisci "num-server-rounds=$R fraction-train=0.5 local-epochs=2 lr-decay=0.97 strategy=\"fedavg\" evaluate-central-test=true"
    esegui_o_fallisci "num-server-rounds=$R fraction-train=0.5 local-epochs=2 lr-decay=0.97 strategy=\"fedprox\" proximal-mu=1.0 evaluate-central-test=true"
    esegui_o_fallisci "num-server-rounds=$R fraction-train=0.5 local-epochs=1 lr-decay=0.97 strategy=\"fedyogi\" evaluate-central-test=true"
    esegui_o_fallisci "num-server-rounds=$R fraction-train=0.5 local-epochs=2 lr-decay=0.97 strategy=\"classaware\" evaluate-central-test=true"
fi
esegui_o_fallisci "num-server-rounds=$R fraction-train=0.5 local-epochs=2 lr-decay=0.97 strategy=\"fednova\" local-optimizer=\"sgd\" evaluate-central-test=true"
esegui_o_fallisci "num-server-rounds=$R fraction-train=0.5 local-epochs=2 lr-decay=0.97 strategy=\"fedavgm\" server-learning-rate=1.0 server-momentum=0.9 evaluate-central-test=true"
esegui_o_fallisci "num-server-rounds=$R fraction-train=0.5 local-epochs=2 lr-decay=0.97 strategy=\"fedadagrad\" server-learning-rate=0.1 server-tau=0.001 evaluate-central-test=true"

if [[ "$RUN_ADDITIONAL_ONLY" != "1" ]]; then
    prepara_dati --mini-dataset "$SHARED_PER_CLASS"
    esegui_o_fallisci "num-server-rounds=$R fraction-train=0.5 local-epochs=2 lr-decay=0.97 strategy=\"fedavg\" evaluate-central-test=true"
fi

echo "Suite completata. Risultati: $FEDRNN_OUTPUT_ROOT"
