# CLIENT APP
# Strutturato con 3 handler:
#   1. risposta alla richiesta di statistiche che il server manda la prima volta
#   2. e 3. sono le due metà di ciascun round, cioè addestrare e valutare il modello globale sui dati locali
# I dati visibili sono solo quelli del proprio shard 

from __future__ import annotations

import numpy as np
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fedrnn import config as cfg
from fedrnn.data import build_client_loaders, client_training_statistics, load_meta
from fedrnn.strategies import TRAIN_CLASS_COUNTS_KEY
from fedrnn.task import DeviceRNN, evaluate as evaluate_fn, flatten_confusion
from fedrnn.task import local_train, pick_device

app = ClientApp()

# DIVISIONE DATI
# client e shard sono collegati tramite il partition-id (0 - num-partition-1) assegnato da flwr ad ogni SuperNode

# prima di proseguire viene verificato che il numero di SuperNode della simulazione e numero shard coincidano per evitare che
# o i file non vengano trovati (più SuperNode che shard) o il dataset non sia completamente utilizzato (più shard che SuperNode)

def _partition_id(context: Context) -> int:
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    meta = load_meta()
    if num_partitions != meta["num_partitions"]:
        raise RuntimeError(
            f"La federazione ha {num_partitions} SuperNode ma in {cfg.SHARD_ROOT} "
            f"ci sono {meta['num_partitions']} shard. Porta "
            f"num-supernodes nella configurazione della simulazione a "
            f"{meta['num_partitions']}, "
            "oppure rigenera gli shard con prepare_data.py."
        )
    return partition_id


def _model_from_message(msg: Message) -> DeviceRNN:
    # costruisce il modello e carica dentro i pesi arrivati dal server
    model = DeviceRNN()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    return model


def _config_array(msg: Message, key: str, expected_len: int) -> np.ndarray:
    # legge dal ConfigRecord un vettore di float mandato dal server
    values = msg.content["config"][key]
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (expected_len,):
        raise RuntimeError(
            f"Il server ha mandato {key!r} con {array.shape[0]} valori invece di "
            f"{expected_len}."
        )
    return array

# STATISTICHE
# 1. il client, prima di addestrare, su richiesta del server, invia quanti valori ha per feature, la loro somma, 
#    la somma dei quadrati e quanti campioni ha per classe

# 2. successivamente il server calcola media, dev std e i pesi per classe per la funzione di costo per avere uno scaler su tutta la federazione
#    e mantenendo i dati dentro i client

# uso decoratore query per effettuare scambio che non è di train o valutazione
@app.query()
def query(msg: Message, context: Context) -> Message:
    # Restituisce le statistiche aggregate dei dati locali di addestramento
    partition_id = _partition_id(context)
    stats, y = client_training_statistics(partition_id)

    metrics: dict[str, float | list[float]] = dict(stats)
    metrics["class_count"] = [float((y == idx).sum()) for idx in range(cfg.NUM_CLASSES)]
    # num-examples è la chiave con cui la strategia pesa le medie. 
    # qui non pesa niente, ma tenerla presente in tutti i MetricRecord evita di avere risposte
    # con chiavi diverse da un tipo di messaggio all'altro
    metrics["num-examples"] = float(len(y))

    content = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)


# ADDESTRAMENTO LOCALE

@app.train()
def train(msg: Message, context: Context) -> Message:
    # viene assegnato il partition id e il device usato (cpu/gpu)
    partition_id = _partition_id(context)
    device = pick_device()

    # inizializzazione del modello e delle config del server
    model = _model_from_message(msg)
    conf = msg.content["config"]

    # inizializzazione degli array della media, dev std e pesi delle classi inviati dal server
    # ^ vengono inviati dal server perchè se avessi pesi locali e distribuzioni diverse allora avremmo funzioni di costo diverso e la media
    # dei pesi non avrebbe senso
    mean = _config_array(msg, "feature-mean", cfg.NUM_FEATURES)
    std = _config_array(msg, "feature-std", cfg.NUM_FEATURES)
    class_weights = torch.from_numpy(
        _config_array(msg, "class-weights", cfg.NUM_CLASSES)
    )

    # loader dati di addestramento 
    train_loader, _, y_train = build_client_loaders(
        partition_id,
        batch_size=int(context.run_config["batch-size"]),
        mean=mean,
        std=std,
    )

    # per il lr, il server manda il valore iniziale e il fattore di decadimento poi flwr inserisce da solo il numero round dentro il config
    # la formula del decadimento è lr * decay^(round-1), dove al primo round = lr

    # NOTA: proximal-mu c'è solo quando uso FedProx come strategia, quindi nelle altre resta a 0
    server_round = int(conf.get("server-round", 1))
    lr_round = float(conf["lr"]) * float(conf.get("lr-decay", 1.0)) ** max(
        0, server_round - 1
    )

    # addestramento 
    stats = local_train(
        model,
        train_loader,
        epochs=int(context.run_config["local-epochs"]),
        lr=lr_round,
        weight_decay=float(context.run_config["weight-decay"]),
        class_weights=class_weights,
        device=device,
        optimizer_name=str(context.run_config["local-optimizer"]),
        proximal_mu=float(conf.get("proximal-mu", 0.0)),
    )

    # array metriche da inviare
    metrics = {
        "train_loss": stats["train_loss"],
        "num_batches": stats["num_batches"],
        "num-examples": float(len(y_train)),
        # Identificatore anonimo e stabile dello shard. Il server lo salva solo
        # per costruire distribuzioni/heatmap di loss, mai insieme ai dati.
        "client-id": float(partition_id),
        # Necessario solo a ClassAwareFedAvg: sono conteggi dell'insieme di
        # training (la validation locale è già stata esclusa da y_train).
        TRAIN_CLASS_COUNTS_KEY: [
            float((y_train == class_idx).sum()) for class_idx in range(cfg.NUM_CLASSES)
        ],
        "server-round": float(server_round),
        "lr": lr_round,
    }

    # costruzione del body del messaggio da inviare al server
    content = RecordDict(
        {
            "arrays": ArrayRecord(model.state_dict()),
            "metrics": MetricRecord(metrics),
        }
    )
    # rimando indietro i pesi e il numero di campioni usati
    return Message(content=content, reply_to=msg)


# VALUTAZIONE LOCALE
# il client effettua una valutazione del modello globale sui dati di validazione che non ha utilizzato per il train e restituisce la loss e 
# la matrice di confusione

# questa sarebbe l'unica "valutazione" realizzabile in un modello reale perchè il test set centralizzato non esiste

# in questo caso, per confrontare il modello federato con quello centralizzato, il test set viene tenuto per vedere quanto costa federare

@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    partition_id = _partition_id(context)
    device = pick_device()

    model = _model_from_message(msg)
    conf = msg.content["config"]

    mean = _config_array(msg, "feature-mean", cfg.NUM_FEATURES)
    std = _config_array(msg, "feature-std", cfg.NUM_FEATURES)
    class_weights = torch.from_numpy(
        _config_array(msg, "class-weights", cfg.NUM_CLASSES)
    )

    _, val_loader, _ = build_client_loaders(
        partition_id,
        batch_size=int(context.run_config["batch-size"]),
        mean=mean,
        std=std,
    )

    loss, confusion, n_samples = evaluate_fn(
        model, val_loader, class_weights=class_weights, device=device
    )

    metrics = {
        "eval_loss": loss,
        "confusion": flatten_confusion(confusion),
        "num-examples": float(n_samples),
        "server-round": float(conf.get("server-round", 0)),
    }
    content = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)
