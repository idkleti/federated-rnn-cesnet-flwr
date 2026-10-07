#!/usr/bin/env bash

# Completa la valutazione finale non-IID in tools/esperimenti_finali.sh con le
# stesse strategie e gli stessi iperparametri sulle due baseline IID:
#   - random: client IID di uguale dimensione;
#   - random-sizes: IID ma con le dimensioni client delle subnet reali.
#
# Non ripete le 20 run non-IID gia' prodotte: al termine, grafici.py trova
# 4 configurazioni subnet + 4 random + 4 random-sizes, tutte con 5 repliche.
# Esegui esclusivamente via SLURM:
#   sbatch esperimenti_copernico.sh tools/esperimenti_finali_iid.sh

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd -- "$script_dir/.."

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

esegui_configurazioni_selezionate() {
    # Parametri selezionati dalla validation-only, identici alle run subnet
    # finali. Il test e' eseguito una sola volta dopo tale selezione.
    esegui "num-server-rounds=$R fraction-train=1.0 local-epochs=2 lr-decay=0.97 strategy=\"fedavg\" evaluate-central-test=true"
    esegui "num-server-rounds=$R fraction-train=0.5 local-epochs=2 lr-decay=0.97 strategy=\"fedprox\" proximal-mu=1.0 evaluate-central-test=true"
    esegui "num-server-rounds=$R fraction-train=1.0 local-epochs=1 lr-decay=0.97 strategy=\"fedyogi\" evaluate-central-test=true"
    esegui "num-server-rounds=$R fraction-train=0.5 local-epochs=2 lr-decay=0.97 strategy=\"classaware\" evaluate-central-test=true"
}

for partition in random random-sizes; do
    echo "=== IID partition: $partition ==="
    prepara_dati --partition "$partition"
    esegui_configurazioni_selezionate
done

# Non modifica gli history gia' scritti; ripristina solo gli shard di lavoro
# non-IID perche' figura 05 di grafici.py legge anche shards/meta.json.
prepara_dati

# Produce il confronto completo e aggiorna tables/ e figure_tables.zip. Le run
# validation-only restano escluse perche' non hanno metriche di test.
python tools/grafici.py
