#!/usr/bin/env bash
# Ablazione non-IID: FedAvg e FedProx (mu=1) con 1, 2, 4 epoche locali.
# Ogni configurazione: 25 round, partecipazione 0.5, decay 0.97, Adam.
# Cinque repliche nuove per configurazione, anche per la baseline a 2 epoche.
# Risultati isolati in outputs/ablazione_epoche per non alterare le figure finali.
# Avviare esclusivamente via: sbatch esperimenti_ablazione_epoche.sbatch

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Questa ablazione deve essere eseguita su un nodo di calcolo tramite sbatch." >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd -- "$script_dir/.."

FEDERATION_CONFIG="num-supernodes=69 client-resources-num-cpus=11 client-resources-num-gpus=0.5"
export FEDRNN_OUTPUT_ROOT="$PWD/outputs/ablazione_epoche"
mkdir -p "$FEDRNN_OUTPUT_ROOT"
echo "Output ablazione: $FEDRNN_OUTPUT_ROOT"

# Il wrapper SLURM imposta FEDRNN_OFFLINE=1 e verifica il dataset locale.
python prepare_data.py --offline

for strategy in fedavg fedprox; do
    for local_epochs in 1 2 4; do
        run_config="num-server-rounds=25 fraction-train=0.5 local-epochs=$local_epochs lr-decay=0.97 strategy=\"$strategy\" evaluate-central-test=true"
        if [[ "$strategy" == "fedprox" ]]; then
            run_config+=" proximal-mu=1.0"
        fi

        echo "==> $run_config"
        for replicate in 1 2 3 4 5; do
            printf '  [%d/5] %s\n' "$replicate" "$(date +%H:%M:%S)"
            flwr run . --federation-config "$FEDERATION_CONFIG" \
                --run-config "$run_config" --stream
        done
    done
done

echo "Ablazione completata: 30 run; history e checkpoint in $FEDRNN_OUTPUT_ROOT"
