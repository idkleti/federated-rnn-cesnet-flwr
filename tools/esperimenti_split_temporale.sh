#!/usr/bin/env bash

# Split temporale: gli stessi indirizzi in periodi diversi, con le proporzioni dello split per indirizzi applicate ai 280 giorni (80/10/10):
# giorni 1-224 per l'addestramento, 225-252 per la validazione federata, 253-280 per il test del server.
# Una run da 25 round per ognuna delle strategie della suite finale, con gli stessi iperparametri di esperimenti_finali_split_45_10_45.sh.
# Shard in shards/temporale e run in outputs/split_temporale, separati da quelli con lo split per indirizzi.
#
# Avvio, dalla cartella del progetto con il venv attivo:
#       bash tools/esperimenti_split_temporale.sh
#       N=5 bash tools/esperimenti_split_temporale.sh      (5 run per strategia)
#       R=2 bash tools/esperimenti_split_temporale.sh      (prova veloce con 2 round)

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd -- "$script_dir/.."

export FEDRNN_SHARD_ROOT="$PWD/shards/temporale"
export FEDRNN_OUTPUT_ROOT="$PWD/outputs/split_temporale"
mkdir -p "$FEDRNN_SHARD_ROOT" "$FEDRNN_OUTPUT_ROOT"

# Flower usa le dipendenze già installate nel venv invece di ricreare un ambiente con uv sync a ogni run (in locale restava fermo per minuti)
export FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1

R="${R:-25}"
N="${N:-1}"
FEDERATION_CONFIG="${FEDERATION_CONFIG:-client-resources-num-cpus=8 client-resources-num-gpus=0.5}"

# il superlink locale resta vivo fra una run e l'altra e le app che lancia leggono le cartelle dalla shell che lo ha avviato:
# lo fermo prima, così riparte con le cartelle di questo script, e alla fine, così le run normali non le ereditano
ferma_superlink() {
    if pgrep -f flower-superlink > /dev/null; then
        echo "Fermo il superlink locale rimasto attivo"
        pkill -f flower-superlink || true
        while pgrep -f flower-superlink > /dev/null; do sleep 1; done
    fi
}

prepara_dati() {
    python prepare_data.py --split temporal "$@"

    # con tutti gli indirizzi nei client le subnet sopra la soglia sono 71 invece di 69: il numero di SuperNode segue gli shard appena generati
    local num_supernodes
    num_supernodes="$(python -c 'import json; from fedrnn import config; print(json.loads(config.meta_path().read_text(encoding="utf-8"))["num_partitions"])')"
    FEDERATION_CONFIG="num-supernodes=$num_supernodes $FEDERATION_CONFIG"
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
            ferma_superlink
            exit 1
        fi
        printf '%s\n' "$output" | grep -oE 'macro-F1 sul test, checkpoint selezionato: [0-9.]+' | tail -1 || true
    done
}

ferma_superlink
prepara_dati
esegui "num-server-rounds=$R fraction-train=0.5 local-epochs=2 lr-decay=0.97 strategy=\"fedavg\" evaluate-central-test=true"
esegui "num-server-rounds=$R fraction-train=0.5 local-epochs=2 lr-decay=0.97 strategy=\"fedprox\" proximal-mu=1.0 evaluate-central-test=true"
esegui "num-server-rounds=$R fraction-train=0.5 local-epochs=1 lr-decay=0.97 strategy=\"fedyogi\" evaluate-central-test=true"
esegui "num-server-rounds=$R fraction-train=0.5 local-epochs=2 lr-decay=0.97 strategy=\"classaware\" evaluate-central-test=true"
esegui "num-server-rounds=$R fraction-train=0.5 local-epochs=2 lr-decay=0.97 strategy=\"fednova\" local-optimizer=\"sgd\" evaluate-central-test=true"
esegui "num-server-rounds=$R fraction-train=0.5 local-epochs=2 lr-decay=0.97 strategy=\"fedavgm\" server-learning-rate=1.0 server-momentum=0.9 evaluate-central-test=true"
esegui "num-server-rounds=$R fraction-train=0.5 local-epochs=2 lr-decay=0.97 strategy=\"fedadagrad\" server-learning-rate=0.1 server-tau=0.001 evaluate-central-test=true"
ferma_superlink

echo "Fatto. Storici delle run in $FEDRNN_OUTPUT_ROOT"
