# Costanti e percorsi condivisi da tutto il progetto 

from __future__ import annotations

import os
from pathlib import Path

# INDIVIDUAZIONE DI DOVE SI TROVANO DATI, SHARD E OUTPUT rispetto ai vari moduli

# Quando si lancia `flwr run .`, Flower impacchetta il progetto e lo esegue da una copia dentro
#       `~/.flwr/apps/<nome>.<versione>.<hash>/`. 
# In questa copia non ci sono gli shard, quindi scrivere solamente __file__ non va bene perchè mi troverei all'interno della copia.

# Di conseguenza dobbiamo "trovare" dove si trovano la cartella time_dataset/, shards/ e outputs/ rispetto al progetto originale.
# L'ordine di ricerca è: 
#   1. la variabile d'ambiente, se c'è 
#   2. directory da cui è stato lanciato il comando, che quando si esegue `flwr run .` dalla  cartella del progetto è quella giusta
#   3. posizione di questo file, che è quella corretta per `prepare_data.py` e per qualsiasi script eseguito direttamente

# calcola la radice del progetto a partire dal percorso di config.py
_PROJECT_ROOT = Path(__file__).resolve().parents[1] #parents[0] sarebbe fedrnn/

def _resolve(env_var: str, folder: str) -> Path:
    # radice di una cartella di dati, cercata nell'ordine descritto sopra

    # var ambiente
    from_env = os.getenv(env_var)
    if from_env:
        return Path(from_env)

    # dir da cui ho lanciato il comando
    from_cwd = Path.cwd() / folder
    if from_cwd.exists():
        return from_cwd

    # posizione di questo file
    return _PROJECT_ROOT / folder

# salvo dove si trovano le posizioni
DATA_ROOT = _resolve("CESNET_DATA_ROOT", "time_dataset")
SHARD_ROOT = _resolve("FEDRNN_SHARD_ROOT", "shards")
OUTPUT_ROOT = _resolve("FEDRNN_OUTPUT_ROOT", "outputs")

# metadati del file .json per la creazione delle shard
SHARD_META_FILE = "meta.json"
# shard del server
SERVER_SHARD_FILE = "server_test.npz"

# CREAZIONE DEI PATH DEGLI SHARD
# creazione del path per ogni client shard
def client_shard_path(partition_id: int) -> Path:
    # metto 4 cifre alla fine per mantenerli ordinati
    return SHARD_ROOT / f"client_{partition_id:04d}.npz"

# creazione del path per lo shard server e gli indirizzi test
# normalmente non ci dovrebbe essere perchè il server non può tenere una parte dei dati di tutti, ma in questo caso voglio confrontare 
# con la versione centralizzata per vedere quanto costa federare
def server_shard_path() -> Path:
    return SHARD_ROOT / SERVER_SHARD_FILE

# path dei metadati delle shard
def meta_path() -> Path:
    return SHARD_ROOT / SHARD_META_FILE


# FORMA DI UN CAMPIONE

# Nel dataset CESNET-TimeSeries24, una serie temporale corrisponde a un indirizzo IP, quindi a un dispositivo

# tupla con questo ordine per rispettare l'ordine di features_to_take="all" di TS-Zoo
FEATURE_NAMES: tuple[str, ...] = (
    "n_flows",
    "n_packets",
    "n_bytes",
    "sum_n_dest_asn",
    "avg_n_dest_asn",
    "std_n_dest_asn",
    "sum_n_dest_ports",
    "avg_n_dest_ports",
    "std_n_dest_ports",
    "sum_n_dest_ip",
    "avg_n_dest_ip",
    "std_n_dest_ip",
    "tcp_udp_ratio_packets",
    "tcp_udp_ratio_bytes",
    "dir_ratio_packets",
    "dir_ratio_bytes",
    "avg_duration",
    "avg_ttl",
)
NUM_FEATURES = len(FEATURE_NAMES)
# 280 giorni
SEQ_LEN = 280

# CLASSI 

# L'annotazione device_type_ip_address_full, distribuita insieme a TS-Zoo, etichetta 92.379 dei 275.124 indirizzi disponibili tramite la colonna group
# mentre gli altri restano senza etichetta e vengono scartati.

# uso lo stesso ordine di LabelEncoder di scikit learn che ordina alfabeticamente altrimenti gli ordini non corrispondono più
CLASS_NAMES: tuple[str, ...] = ("end-device", "net-device", "server")
NUM_CLASSES = len(CLASS_NAMES)

# SPLIT E RIPRODUCIBILITA'

# 111 è il seed del progetto centralizzato e 80/10/10 sono le sue proporzioni per avere come indirizzi di test gli stessi 9238 del modello centralizzato.
# Il modello federato viene quindi misurato sugli stessi dispositivi del baseline, e la differenza fra le
# due macro-F1 si può attribuire al federated learning invece che a un test set capitato più facile.

RANDOM_STATE = 111
TEST_RATIO = 0.10
VAL_RATIO = 0.10

# Quanto di ogni shard il client tiene da parte per valutare il modello globale
CLIENT_VAL_FRACTION = 0.20
