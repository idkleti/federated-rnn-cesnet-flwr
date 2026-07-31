
# Accesso dati da lato client e server leggendo gli shard simulando una separazione fisica vera

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, TensorDataset

from fedrnn import config

# leggo uno shard e tengo una copia in memoria

# eseguo dei controlli rigorosi della memoria perchè utilizzando la partizione per subnet il client più grande riceve migliaia
# di indirizzi e la memoria si riempirebbe velocemente visto che verrebbero tenuti in memoria la versione dell'array grezza, standardizzata, 
# sottoinsieme di addestramento e validazione (causato dal fatto che array di indici in numpy o pytorch copia i dati e non restituisce una vista)

# flwr usa lo stesso processo per client diversi lungo i round (problema di una cache senza limiti) quindi tengo solo lo shard corrente al costo
# di una rilettura da disco quando il processo cambia client per mettere un limite al consumo di memoria

# tengo in cache l'array già standardizzato invece di quello grezzo visto che media e dev std sono fissate una sola volta e anche perchè
# i DataLoader usano Subset per tenere una lista di indici e non duplicare i tensori
_CacheEntry = tuple[np.ndarray, np.ndarray, np.ndarray]
_cache: dict[tuple, _CacheEntry] = {}


def _stats_key(mean: np.ndarray | None, std: np.ndarray | None) -> tuple:
    """Firma compatta delle statistiche, per invalidare la cache se cambiano."""
    if mean is None or std is None:
        return ("raw",)
    return (
        "std",
        float(np.asarray(mean, dtype=np.float64).sum()),
        float(np.asarray(std, dtype=np.float64).sum()),
    )

# z-score sull'array appena letto, in place, prima di metterlo in cache.
def load_shard(
    path: Path,
    *,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> _CacheEntry:
    # carica (X, y, ts_ids) già standardizzato se mean e std sono forniti
    key = (str(path), _stats_key(mean, std))
    if key in _cache:
        return _cache[key]

    if not path.exists():
        raise FileNotFoundError(
            f"Shard mancante: {path}\nEsegui prima `python prepare_data.py`."
        )

    with np.load(path) as data:
        X = data["X"].astype(np.float32, copy=False)
        y = data["y"].astype(np.int64, copy=False)
        ts_ids = data["ts_ids"]

    # TS-zoo riempie in automatico i buchi con il valore default della feature, però effettuo un controllo sulla presenza di valori NaN a prescindere
    # per evitare che la loss diventi NaN
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # effettuo lo z-score per portare ogni metrica sulla scala (perchè ho n_bytes = 25000000 e dir_ratio_packets = 0.6 quindi n_bytes pesa di più)
    if mean is not None and std is not None:
        # con -= e /= io NON creo un nuovo array ma lavoro su quello già esistente quindi non ho mai due copie dell'array in memoria
        X -= np.asarray(mean, dtype=np.float32).reshape(1, -1, 1) 
        X /= np.asarray(std, dtype=np.float32).reshape(1, -1, 1)

    # svuoto la cache prima di passare al client dopo
    _cache.clear()
    _cache[key] = (X, y, ts_ids)
    return X, y, ts_ids


_meta_cache: dict[str, dict] = {}


def load_meta() -> dict:
    # metadati scritti da prepare_data.py

    # vengono piazzati in cache perchè la clientapp li rilegge ad ogni messaggio per controllare che il proprio partition-id sia coerente con il
    # numero di shard 
    path = config.meta_path()
    key = str(path)
    if key in _meta_cache:
        return _meta_cache[key]

    if not path.exists():
        raise FileNotFoundError(
            f"Metadati mancanti: {path}\nEsegui prima `python prepare_data.py`."
        )
    _meta_cache[key] = json.loads(path.read_text(encoding="utf-8"))
    return _meta_cache[key]


# divisione stratificata dello shard in addestramento e validazione quando è possibile altrimenti split casuale (si verifica quando una classe ha
# meno di due elementi, la validazione non ha almeno un elemento di ciascuna classe presente, l'addestramento non ha almeno un elemento di ciascuna 
# classe presente)

def client_train_val_split(
    y: np.ndarray,
    *,
    val_fraction: float,
    partition_id: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Indici di addestramento e di validazione locali per un client."""
    n = len(y)
    indices = np.arange(n)

    n_val = int(round(n * val_fraction))
    if n_val < 1 or n - n_val < 1:
        # shard troppo piccolo quindi si allena e si valuta sugli stessi dati, tuttavia pesa poco perchè ha pochi campioni
        return indices, indices

    seed = random_state + partition_id
    counts = np.bincount(y)
    presenti = counts[counts > 0]
    n_classi = len(presenti)

    stratificabile = (
        presenti.min() >= 2          # ogni classe ha almeno due esemplari
        and n_val >= n_classi        # la validazione può contenerle tutte
        and (n - n_val) >= n_classi  # e anche l'addestramento
    )

    train_idx, val_idx = train_test_split(
        indices,
        test_size=n_val,
        random_state=seed,
        stratify=y if stratificabile else None,
    )
    return np.sort(train_idx), np.sort(val_idx)


# statistiche locali per standardizzazione federata

# scrivo media e varianza come funzioni di tre somme (numero di valori, somma di valori, somma di quadrati) in modo tale che ogni client le calcoli
# sul proprio addestramento locale e il server le sommi, ottenendo la stessa media e dev std di uno scaler globale

# la somma del traffico non permette di risalire al traffico di un singolo dispositivo
def local_feature_statistics(
    X: np.ndarray,
    indices: np.ndarray | None = None,
    *,
    block: int = 256,
) -> dict[str, list[float]]:
    # numero di valori e somme aggregati come float64

    # indices seleziona la serie da usare senza creare una copia in memoria
    rows = np.arange(X.shape[0]) if indices is None else np.asarray(indices)
    n_features = X.shape[1]

    total = np.zeros(n_features, dtype=np.float64)
    total_sq = np.zeros(n_features, dtype=np.float64)

    # eseguo la somma a blocchi per non occupare troppa memoria
    # somma quadrati in float64 altrimento perderei le cifre più basse
    for start in range(0, len(rows), block):
        chunk = X[rows[start : start + block]].astype(np.float64)
        total += chunk.sum(axis=(0, 2))
        total_sq += (chunk**2).sum(axis=(0, 2))

    n_values = float(len(rows) * X.shape[2])  # serie per passi temporali
    return {
        "stat_count": [n_values] * n_features,
        "stat_sum": total.tolist(),
        "stat_sum_sq": total_sq.tolist(),
    }


def statistics_to_mean_std(
    count: np.ndarray,
    total: np.ndarray,
    total_sq: np.ndarray,
    *,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    # calcolo media e dev std per feature a partire dalle somme
    count = np.maximum(np.asarray(count, dtype=np.float64), 1.0)
    mean = np.asarray(total, dtype=np.float64) / count
    variance = np.asarray(total_sq, dtype=np.float64) / count - mean**2
    std = np.sqrt(np.maximum(variance, 0.0))
    return mean.astype(np.float32), np.maximum(std, eps).astype(np.float32)


# shard diviso in addestramento e validazione locale, standardizzato e impacchettato in 2 DataLoader
def build_client_loaders(
    partition_id: int,
    *,
    batch_size: int,
    mean: np.ndarray,
    std: np.ndarray,
    val_fraction: float = config.CLIENT_VAL_FRACTION,
    random_state: int = config.RANDOM_STATE,
) -> tuple[DataLoader, DataLoader, np.ndarray]:
    # recupero delle etichette per comunicare al server quanti campioni ha davvero usato 
    X, y, _ = load_shard(config.client_shard_path(partition_id), mean=mean, std=std)

    train_idx, val_idx = client_train_val_split(
        y,
        val_fraction=val_fraction,
        partition_id=partition_id,
        random_state=random_state,
    )

    # Subset tiene solo la lista degli indici quindi rimane in memoria una copia sola dello shard nella cache
    # from_numpy condivide la memoria con l'array in cache anzichè copiarlo 
    full = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    train_ds = Subset(full, train_idx.tolist())
    val_ds = Subset(full, val_idx.tolist())


    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False, # dato che ci sono client con poche decine di indirizzi, non scarto l'ultimo batch incompleto per evitare che non
            # si allenino affatto
        num_workers=0, # il dataloader di PyTorch avvia un processo per worker e visto che siamo all'interno del processo flwr per il client
            # e i dati sono già tutti in memoria come tensori, non c'è bisogno di worker perchè non ho operazioni I/O da effettuare
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(batch_size, 128),
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader, y[train_idx]

# invio delle statistiche iniziali in clientapp da @app.query()
def client_training_statistics(
    partition_id: int,
    *,
    val_fraction: float = config.CLIENT_VAL_FRACTION,
    random_state: int = config.RANDOM_STATE,
) -> tuple[dict[str, list[float]], np.ndarray]:
    # uso solo la parte di addestramento
    X, y, _ = load_shard(config.client_shard_path(partition_id))
    train_idx, _ = client_train_val_split(
        y,
        val_fraction=val_fraction,
        partition_id=partition_id,
        random_state=random_state,
    )
    return local_feature_statistics(X, train_idx), y[train_idx]


# test set del server

# utilizzati solamente per effettuare il confronto con la macro-f1 score del modello centralizzato
# NOTA: questo esiste solo perchè voglio effettuare un confronto dato che normalmente si verifica come va il modello con le statistiche inviate dal
# client (vedi @app.evaluate() in clientapp)

def build_server_test_loader(
    *,
    batch_size: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> DataLoader:
    X, y, _ = load_shard(config.server_shard_path(), mean=mean, std=std)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
