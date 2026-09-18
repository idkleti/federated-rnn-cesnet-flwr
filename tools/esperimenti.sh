#!/usr/bin/env bash

# Questo file contiene le run mostrate in RISULTATI.md
# Tempo totale di addestramento per la mia macchina: 47 ore

# Eseguibile con:
#     source venv/bin/activate  # versione python3.12 preferibilmente
#     bash tools/esperimenti.sh
# solo dopo aver scaricato il dataset ed eventualmente caricato le shard (python prepare_data.py)
# NOTA: ad ogni esecuzione vengono aggiunti nuovi file in ../outputs/

# Per evitare di dover lasciare in esecuzione per 47 ore, commentare i blocchi che non servono e lanciarne uno per volta

# Nel progetto le configurazioni vengono ripetute 25 volte poichè con solo un paio di esecuzioni non è possibile distinguere tra coincidenza e risultati veri

set -euo pipefail

# BASH_SOURCE resta il percorso dello script anche se viene eseguito con `source`.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd -- "$script_dir/.." # Flower deve essere eseguito accanto a pyproject.toml

# Per retrocompatibilità il comportamento predefinito resta invariato. Sul
# cluster, FEDRNN_OFFLINE=1 aggiunge --offline a ogni rigenerazione degli shard.
prepare_data_args=()
if [[ "${FEDRNN_OFFLINE:-0}" == "1" ]]; then
    prepare_data_args=(--offline)
fi

prepara_dati() {
    python prepare_data.py "${prepare_data_args[@]}" "$@"
}

R=50   # round per le run normali
N=25   # quante volte ripetere ogni configurazione

esegui() {
    echo "==> $*"
    for i in $(seq 1 $N); do
        printf '  [%2d/%2d] %s  ' "$i" "$N" "$(date +%H:%M:%S)"
        if ! output=$(flwr run . --run-config "$*" --stream 2>&1); then
            printf '%s\n' "$output" >&2
            return 1
        fi
        printf '%s\n' "$output" | grep -oE "checkpoint selezionato: [0-9.]+" | tail -1 || true
    done
}


# RUN EFFETTUATE in maniera "realistica" (dati divisi per subnet)
prepara_dati

# 1. il numero di client che partecipano viene modificato al 30%, 50% e 100%
esegui "num-server-rounds=$R fraction-train=0.3 lr-decay=1.0 strategy=\"fedavg\""
esegui "num-server-rounds=$R fraction-train=0.5 lr-decay=1.0 strategy=\"fedavg\""
esegui "num-server-rounds=$R fraction-train=1.0 lr-decay=1.0 strategy=\"fedavg\""


# 2. learning rate con decadimento
esegui "num-server-rounds=$R fraction-train=0.5 lr-decay=0.97 strategy=\"fedavg\""


# 3. strategie alternative al punto 2 con stessa partecipazione del 50% e decadimento learning rate
esegui "num-server-rounds=$R fraction-train=0.5 lr-decay=0.97 strategy=\"fedadam\""
esegui "num-server-rounds=$R fraction-train=0.5 lr-decay=0.97 strategy=\"fedprox\"" # usa proximal-mu=0.1 di default


# 4. 100 round anzichè 50
#esegui "num-server-rounds=100 fraction-train=0.5 lr-decay=1.0 strategy=\"fedavg\""

# 5. cambiamento del valore di proximal-mu di fedprox
# QUESTA SERIE DI ESECUZIONI E' STATA ESEGUITA DOPO QUELLE DEI PUNTI 1-8
# proximal-mu pesa quanto un client viene penalizzato man mano che si allontana dai pesi che il server gli ha spedito
esegui "num-server-rounds=$R fraction-train=0.5 lr-decay=0.97 strategy=\"fedprox\" proximal-mu=0.2"
esegui "num-server-rounds=$R fraction-train=0.5 lr-decay=0.97 strategy=\"fedprox\" proximal-mu=0.5"
esegui "num-server-rounds=$R fraction-train=0.5 lr-decay=0.97 strategy=\"fedprox\" proximal-mu=0.8"
esegui "num-server-rounds=$R fraction-train=0.5 lr-decay=0.97 strategy=\"fedprox\" proximal-mu=1.0"


# RUN SPERIMENTALI (dati divisi randomicamente, copia della classe minoritaria e mantenimento della grandezza dei client iniziali ma con dati casuali)
# Una volta effettuati i primi 4 punti, la configurazione di addestramento migliore fra quelle provate fino a quel momento (fedavg, partecipazione 0.5, decadimento 0.97)
# viene fissata e l'unica cosa che cambia è come vengono divisi i dati per vedere se si può guadagnare qualcosa e come l'eterogeneità dei client influisce sul modello

# Il progetto è in grado di eseguire tutte le configurazioni reali e sperimentali, l'unica cosa da cambiare è come vengono creati gli shard

# La modalità usata è registrata in shards/meta.json (campi partition e rebalance_net_device), nome file history e campo partition in history

# !! IL TEST SET DEL SERVER NON VIENE MODIFICATO E RIMANE SEMPRE LO STESSO AD OGNI ESECUZIONE 

# 6. indirizzi mescolati a caso con dimensione dei client uguale
prepara_dati --partition random
esegui "num-server-rounds=$R fraction-train=0.5 lr-decay=0.97 strategy=\"fedavg\""


# 7. copia della classe minoritaria (net-device) per rilassare ipotesi del federated learning

# Questa idea è la stessa nel paper Zhao et al. 2018 "Federated Learning with Non-IID Data" dove
# un insieme ristretto e condiviso (in questo caso di net-device) viene distribuito a tutti i partecipanti

# Le serie prestate sono 425 e sono tutte diverse per evitare che il modello impari a memoria le stesse serie ripetute

# Le righe prestate sono in coda allo shard e non entrano nella validazione locale perchè sennò il modello validerebbe
# una serie che potrebbe aver già visto in addestramento in precedenza, aumentando la f1 score senza alcun miglioramento reale

# Con N=10 ricevono copie i 53 client che ne hanno meno di dieci (27 non ne hanno nemmeno uno e 26 che ne hanno fra uno e nove)
prepara_dati --rebalance-net-device 10
esegui "num-server-rounds=$R fraction-train=0.5 lr-decay=0.97 strategy=\"fedavg\""

# 8. indirizzi mescolati con dimensioni delle subnet reali
prepara_dati --partition random-sizes
esegui "num-server-rounds=$R fraction-train=0.5 lr-decay=0.97 strategy=\"fedavg\""

# 9. fedyogi, sulla partizione per subnet come le altre strategie del punto 3
# QUESTA SERIE DI ESECUZIONI E' STATA ESEGUITA DOPO QUELLE SU proximal-mu
prepara_dati
esegui "num-server-rounds=$R fraction-train=0.5 lr-decay=0.97 strategy=\"fedyogi\""


# 10. mini-dataset condiviso
# 500 serie per classe estratte una volta sola e date identiche a tutti i client, quindi duplicate
# QUESTA SERIE DI ESECUZIONI E' STATA ESEGUITA PER ULTIMA, DOPO QUELLE SU fedyogi
prepara_dati --mini-dataset 500
esegui "num-server-rounds=$R fraction-train=0.5 lr-decay=0.97 strategy=\"fedavg\""

# La divisione per istituzione non viene effettuata poichè risulterebbe troppo simile alla divisione per subnet

# per generare le figure:
#     python tools/grafici.py
