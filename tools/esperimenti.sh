#!/usr/bin/env bash

# Questo file contiene le run mostrate in RISULTATI.md
# Tempo totale di addestramento per la mia macchina (specifiche in ../RISULTATI.md): 1h 50m

# Eseguibile con:
#     source venv/bin/activate
#     bash tools/esperimenti.sh
# solo dopo aver caricato gli shard (python prepare_data.py)

# ATTENZIONE: se lo rilanci, vengono aggiunte altre 15 run in outputs/ e i risultati combinati con le nuove run non saranno più gli stessi documentati!!

set -u # fallisce se uso variabile non definita
cd "$(dirname "$0")/.." # mi sposto sopra perchè flwr deve essere eseguito nella stessa dir di pyproject.toml

esegui() {
    echo "==> $*"
    flwr run . --run-config "$*" --stream 2>&1 \
        | grep -E "fedrnn.server" \
        | grep -E "Fine\.|macro-F1 sul test"
}

R=50

# 1. fraction-train = 0.3 e fraction-train = 0.5
esegui "num-server-rounds=$R fraction-train=0.3 lr-decay=1.0 strategy=\"fedavg\""
esegui "num-server-rounds=$R fraction-train=0.3 lr-decay=1.0 strategy=\"fedavg\""

esegui "num-server-rounds=$R fraction-train=0.5 lr-decay=1.0 strategy=\"fedavg\""
esegui "num-server-rounds=$R fraction-train=0.5 lr-decay=1.0 strategy=\"fedavg\""

# 2. fraction-train = 1 (scenario NON realistico)
esegui "num-server-rounds=$R fraction-train=1.0 lr-decay=1.0 strategy=\"fedavg\""
esegui "num-server-rounds=$R fraction-train=1.0 lr-decay=1.0 strategy=\"fedavg\""

# 3. learning rate con decadimento
esegui "num-server-rounds=$R fraction-train=0.5 lr-decay=0.97 strategy=\"fedavg\""
esegui "num-server-rounds=$R fraction-train=0.5 lr-decay=0.97 strategy=\"fedavg\""

# 4. strategie alternative per cercare miglioramenti (vedi RISULTATI.md)
esegui "num-server-rounds=$R fraction-train=0.5 lr-decay=0.97 strategy=\"fedadam\""
esegui "num-server-rounds=$R fraction-train=0.5 lr-decay=0.97 strategy=\"fedadam\""

esegui "num-server-rounds=$R fraction-train=0.5 lr-decay=0.97 strategy=\"fedprox\""
esegui "num-server-rounds=$R fraction-train=0.5 lr-decay=0.97 strategy=\"fedprox\""

# 5. 100 round anzichè 50
esegui "num-server-rounds=100 fraction-train=0.5 lr-decay=1.0 strategy=\"fedavg\""
esegui "num-server-rounds=100 fraction-train=0.5 lr-decay=1.0 strategy=\"fedavg\""

# 6. verifica per FedAdam del server-learning-rate
esegui "num-server-rounds=$R fraction-train=0.5 lr-decay=0.97 strategy=\"fedadam\" server-learning-rate=0.1"
