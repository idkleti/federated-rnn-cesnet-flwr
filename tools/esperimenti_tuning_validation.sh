#!/usr/bin/env bash

# Tuning rigoroso: tutte le decisioni usano soltanto la macro-F1 di validation
# federata. Il test centrale e' disabilitato e non viene neppure caricato.
# Esegui soltanto via SLURM:
#   sbatch esperimenti_copernico.sh tools/esperimenti_tuning_validation.sh

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
        printf '%s\n' "$output" | grep -oE "macro-F1 validation federata [0-9.]+" | tail -1 || true
    done
}

# Medesima partizione naturale e stessi iperparametri base per tutte le run.
# evaluate-central-test=false e' intenzionale: il test resta bloccato fino alla
# scelta finale della configurazione tramite validation.
prepara_dati

# Ablazione pulita: strategia, partecipazione e numero di epoch locali.
for strategy in fedavg fedprox fedyogi classaware; do
    for fraction in 0.5 1.0; do
        for epochs in 1 2; do
            extra=""
            if [[ "$strategy" == "fedprox" ]]; then
                extra=" proximal-mu=0.1"
            fi
            esegui "num-server-rounds=$R fraction-train=$fraction local-epochs=$epochs lr-decay=0.97 strategy=\"$strategy\" evaluate-central-test=false$extra"
        done
    done
done

# Sweep di mu: .1 e' gia' incluso nell'ablazione (f=.5, E=2), quindi non lo
# replichiamo. Tutti gli altri fattori restano f=.5, E=2.
for mu in 0.01 0.05 0.2 0.5 0.8 1.0; do
    esegui "num-server-rounds=$R fraction-train=0.5 local-epochs=2 lr-decay=0.97 strategy=\"fedprox\" proximal-mu=$mu evaluate-central-test=false"
done

# Report separato dal grafico dei risultati sul test: usa soltanto gli storici
# con test_evaluation=disabled e aggiorna anche figure_tables.zip.
python tools/analizza_tuning_validation.py
