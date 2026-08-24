# Costruzione degli shard per i client a partire dal dataset CESNET-TimeSeries24

# da eseguire prima di fare flwr run . :
#   python prepare_data.py

# lo script scarica il dataset se non c'è, ricostruisce lo split del progetto centralizzato e scrive un file .npz per client più uno
# per il test set del server (che in questo caso esiste solo per fare un confronto con il modello centralizzato)

# ogni client corrisponde ad una subnet istituzionale con almeno 10 indirizzi al suo interno

# ogni esecuzione di questo script svuota /shards (tranne test set del server) e riscrive meta.json per indicare con quale modalità sono stati costruiti

# Divisione dei dati
#   --partition subnet : divisione di default del progetto
#   --partition random : suddivisione casuale dei dati in blocchi uguali sullo stesso numero di client della divisione per subnet
#   --partition random-sizes : indirizzi mescolati a caso con dimensioni delle subnet vere
#   --rebalance-net-device N : porta ogni client ad almeno N net-device dando copie a chi ne ha meno;
#       Nella divisione per subnet 27 client su 69 non ne hanno nemmeno uno e altri 26 ne hanno meno di dieci, quindi con N=10 ricevono in 53, 
#       per un totale di 425 serie prestate.
#       Come menzionato in esperimenti.sh e RISULTATI.md, questo argomento rilassa l'ipotesi di federated learning dato che alcune serie passano da un client 
#       all'altro, e l'idea è la stessa di Zhao et al. 2018 in "Federated Learning with Non-IID Data"


# lo script alla fine stampa il numero di options.num-supernodes da inserire in pyproject.toml

# il test set del server non cambia

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from cesnet_tszoo.configs import SeriesBasedConfig
from cesnet_tszoo.datasets import CESNET_TimeSeries24
from cesnet_tszoo.utils.enums import (
    AgreggationType,
    AnnotationType,
    DatasetType,
    FillerType,
    SourceType,
    TimeFormat,
    TransformerType,
)

from fedrnn import config as cfg

log = logging.getLogger("prepare_data")

# quante serie carico in memoria per volta
# get_train_numpy le materializza tutte insieme e lavora in float64, quindi 8000 serie sono circa 320 MB
# (nello shard poi ci finiscono in float32, cioè la metà)
CHUNK_SERIES = 8000

# worker di TS-Zoo per leggere dall'HDF5
WORKERS = 4

# Non filtro sul dataset originale perchè voglio rispettare il modello federato dove ognuno ha i propri dati, mentre con gli shard la separazione
# è proprio fisica e non ha accesso ai dati degli altri client

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # non inserisco --num-clients perchè li ricavo dalle subnet
    p = argparse.ArgumentParser(
        description="Costruisce gli shard per client da CESNET-TimeSeries24.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # aggiunta di argomento facoltativo per modificare la grandezza minima delle subnet (default: 10 indirizzi)
    p.add_argument(
        "--min-subnet-size",
        type=int,
        default=10,
        help="Indirizzi etichettati minimi perché una subnet diventi un client"
    )

    p.add_argument(
        "--partition",
        choices=("subnet", "random", "random-sizes"),
        default="subnet",
        help="Come dividere il pool fra i client"
    )

    p.add_argument(
        "--rebalance-net-device",
        type=int,
        default=0, # default è spento
        metavar="N",
        help="Ogni client con meno di N net-device ne riceve copie dai client che ne hanno di più",
    )

    return p.parse_args(argv)

# Le etichette non si trovano dentro le serie temporali ma all'interno di device_type_ip_address_full (annotazione a parte), che copre solo 
# una parte di tutti gli indirizzi

# tutti quelli senza classificazione di dispositivo e indirizzi non etichettati vengono scartati

# colonna utilizzata: group con valori end-device, net-device e server

def load_labels(dataset) -> pd.Series:
    # Serie pandas indicizzata per ts_id con l'etichetta di ogni indirizzo
    dataset.import_annotations(
        identifier="device_type_ip_address_full", enforce_ids=True
    )
    annotations = dataset.get_annotations(on=AnnotationType.TS_ID)

    ts_id_col = dataset.metadata.ts_id_name
    label_cols = [c for c in annotations.columns if c != ts_id_col]
    if not label_cols:
        raise RuntimeError(
            "Le annotazioni non contengono nessuna colonna di etichette oltre a "
            f"{ts_id_col!r}."
        )
    label_col = "group" if "group" in label_cols else label_cols[0]

    labels = annotations.set_index(ts_id_col)[label_col].dropna()
    log.info("Indirizzi etichettati: %s (colonna %r)", f"{len(labels):,}", label_col)
    return labels


def encode_labels(values: np.ndarray) -> np.ndarray:
    # Da stringhe a interi secondo l'ordine fissato in config.CLASS_NAMES

    # evito il LabelEncoder di scikit-learn di proposito perchè l'ordine delle classi deve coincidere con quello del progetto centralizzato, quindi
    # faccio ordine alfabetico (con cfg.CLASS_NAMES)
    mapping = {name: idx for idx, name in enumerate(cfg.CLASS_NAMES)}
    unknown = set(np.unique(values)) - set(mapping)
    if unknown:
        raise RuntimeError(
            f"Etichette non previste nel dataset: {sorted(unknown)}. "
            f"config.CLASS_NAMES contiene {cfg.CLASS_NAMES}."
        )
    return np.array([mapping[v] for v in values], dtype=np.int64)


# Split di cosa resta al server e cosa va ai client

# per ottenere la stessa divisione del modello centralizzato (80/10/10) effettuo due chiamate a train_test_split con lo stesso seed
# e sullo stesso vettore di etichette nello stesso ordine in modo tale da poter effettuare un confronto di macro-f1 score accurato e vedere
# come va il federated learing rispetto al centralizzato

# gli indirizzi per la federazione sono l'unione degli indirizzi di addestramento e validazione, che verranno poi smistati per subnet istituzionale

def holdout_split(
    ts_ids: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int,
    test_ratio: float,
    val_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Separa il test set del server dal pool che verrà distribuito ai client
    temp_ratio = val_ratio + test_ratio
    train_ids, temp_ids, y_train, y_temp = train_test_split(
        ts_ids,
        y,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=y,
    )
    val_ids, test_ids, y_val, y_test = train_test_split(
        temp_ids,
        y_temp,
        test_size=test_ratio / temp_ratio,
        random_state=random_state,
        stratify=y_temp,
    )

    # indirizzi per client e indirizzi per server
    pool_ids = np.concatenate([train_ids, val_ids])
    pool_y = np.concatenate([y_train, y_val])
    return pool_ids, pool_y, test_ids, y_test

# Scelta di divisione: utilizzo la subnet istituzionale perchè così ho più gruppi rispetto a dividere per istituzione (548 - 283)

# la tabella ids_relationship del dataset collega gli identificatori tra i diversi livelli di aggregazione (quale indirizzo sta in quale
# subnet istituzionale e quale subnet in quale istituzione), tuttavia i nomi esatti delle colonne non sono documentati in modo stabile tra le
# versioni della libreria quindi si cercano anzichè fissarli
def resolve_group_column(relationship: pd.DataFrame, ts_id_col: str) -> str:
    candidates = [c for c in relationship.columns if c != ts_id_col]

    for keyword in ("subnet", "institution"):
        for col in candidates:
            if keyword in col.lower():
                return col

    raise SystemExit(
        "Non riesco a individuare una colonna di raggruppamento in "
        f"ids_relationship. Colonne disponibili: {list(relationship.columns)}. "
    )

# scarto le subnet con meno di min_size indirizzi etichettati 

# la grandezza minima della subnet viene introdotta perchè ci sono subnet che contengono 1/2 indirizzi, e che quindi non hanno un peso rilevante
# all'interno della federazione
def apply_min_subnet_size(
    parts: list[np.ndarray], min_size: int
) -> list[np.ndarray]:
    if min_size <= 1:
        return parts

    tenuti = [p for p in parts if len(p) >= min_size]
    if not tenuti:
        raise SystemExit(
            f"Nessuna subnet raggiunge i {min_size} indirizzi richiesti da "
            "--min-subnet-size. Abbassa la soglia."
        )

    scartate = len(parts) - len(tenuti)
    if scartate:
        persi = sum(len(p) for p in parts) - sum(len(p) for p in tenuti)
        log.info(
            "Soglia a %d indirizzi: tengo %d subnet su %d. Le %d scartate "
            "contenevano %s indirizzi, il %.1f%% del pool.",
            min_size,
            len(tenuti),
            len(parts),
            scartate,
            f"{persi:,}",
            100 * persi / sum(len(p) for p in parts),
        )
    return tenuti

# ogni client racchiude al suo interno una sola intera subnet e vengono ordinati per grandezza in modo decrescente
def partition_by_group(group_ids: np.ndarray) -> list[np.ndarray]:
    # group_ids[i] identifica il gruppo di un campione i e restituisce un array di indici, uno per gruppo, in ordine decrescente
    members: dict[object, list[int]] = defaultdict(list)
    for idx, gid in enumerate(group_ids):
        members[gid].append(idx)

    ordered = sorted(members.values(), key=len, reverse=True)
    return [np.sort(np.asarray(g, dtype=int)) for g in ordered]


# PARTIZIONE CASUALE
# prende in input il num di client e num di indirizzi della partizione per subnet, che vengono divisi in blocchi uguali in base al num di client

# la differenza con random-sizes è che li vengono mantenute le dimensioni vere delle subnet
def partition_random(
    n_addresses: int,
    num_clients: int,
    *,
    seed: int,
    dimensioni: list[int] | None = None,
) -> list[np.ndarray]:
    # mescolo le posizioni del pool e le taglio in num_clients pezzi
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_addresses)

    if dimensioni is None:
        # array_split accetta anche divisioni non esatte: 82.527 su 69 dà pezzi da 1204 e da 1205
        pezzi = np.array_split(perm, num_clients)
    else:
        if sum(dimensioni) != n_addresses:
            raise SystemExit(
                f"Le dimensioni richieste sommano a {sum(dimensioni):,} invece di {n_addresses:,}."
            )
        # np.cumsum dice dove tagliare per ottenere esattamente quelle dimensioni
        pezzi = np.split(perm, np.cumsum(dimensioni)[:-1])

    # ordino gli indici dentro ogni client come fa partition_by_group, così gli shard hanno la stessa
    # forma qualunque sia la modalità, e ordino i client dal più grande al più piccolo per lo stesso motivo
    pezzi = sorted((np.sort(p) for p in pezzi), key=len, reverse=True)
    return [p.astype(int) for p in pezzi]


# RIBILANCIAMENTO DELLA CLASSE MINORITARIA
# restituisce per ogni client la lista di indici dell'insieme di indirizzi che quel client riceve in prestito dagli altri

# le righe in prestito vengono messe in coda allo shard al momento della scrittura e questo permette di tenerli fuori dalla validazione locale

# i donatori di net-device sono quelli che si trovano al di sopra della soglia definita

# la donazione avviene a giro per evitare di copiare sempre le stesse serie o di avere poche serie ripetute moltissime volte
def rebalance_net_device(
    parts: list[np.ndarray],
    pool_y: np.ndarray,
    minimo: int,
    *,
    seed: int,
) -> list[np.ndarray]:
    idx_net = cfg.CLASS_NAMES.index("net-device")

    # quanti net-device ha ogni client e quali sono, indice per indice
    net_di = [p[pool_y[p] == idx_net] for p in parts]
    quanti = np.array([len(v) for v in net_di])

    # chi può donare
    donatori = [i for i in range(len(parts)) if quanti[i] > minimo]
    if not donatori:
        raise SystemExit(
            f"Nessun client ha più di {minimo} net-device, quindi non c'è niente da cui copiare."
        )

    # metto in fila tutte le serie donabili, mescolate, e le distribuisco a giro
    rng = np.random.default_rng(seed)
    disponibili: list[int] = []
    for i in donatori:
        surplus = net_di[i][minimo:]  # i primi `minimo` restano al donatore
        disponibili.extend(int(v) for v in surplus)
    rng.shuffle(disponibili)

    prestiti: list[np.ndarray] = []
    posizione = 0
    for i in range(len(parts)):
        mancanti = max(0, minimo - int(quanti[i]))
        presi: list[int] = []
        for _ in range(mancanti):
            # se la fila finisce si riparte da capo: le serie si ripetono, ma sparse su client diversi
            presi.append(disponibili[posizione % len(disponibili)])
            posizione += 1
        prestiti.append(np.array(presi, dtype=int))

    ricevuti = sum(len(p) for p in prestiti)
    riceventi = sum(1 for p in prestiti if len(p) > 0)
    log.info(
        "Ribilanciamento a %d net-device: %d client ne ricevono %s copie in tutto "
        "(il %.2f%% del pool), attingendo da %d donatori.",
        minimo,
        riceventi,
        f"{ricevuti:,}",
        100 * ricevuti / sum(len(p) for p in parts),
        len(donatori),
    )
    return prestiti


# carico le serie grezze a blocchi

# faccio a blocchi perchè get_train_numpy carica tutto in memoria
# le 83,141 serie del pool sono 3.1 GB in float64, che diventano il doppio nel momento in cui le impila una sull'altra, e su 7.4 GB di RAM non ci stanno
def load_raw_series(
    dataset,
    ts_ids: np.ndarray,
    *,
    workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Carica le serie richieste. Restituisce `(X di forma (n, F, T), ts_ids)`.

    # impostazioni di configurazione della serie e applicazione delle config al dataset 
    series_config = SeriesBasedConfig(
        time_period="all",
        train_ts=[int(i) for i in ts_ids],
        val_ts=None,
        test_ts=None,
        features_to_take="all",
        include_time=False,
        include_ts_id=False,
        time_format=TimeFormat.ID_TIME,
        train_batch_size=64,
        fill_missing_with=FillerType.NO_FILLER,
        transform_with=TransformerType.NO_TRANSFORMER, # uso NO_TRANSFORMER perchè TS-Zoo farebbe la standardizzazione delle feature
            # con lo scaler su tutte le serie messe assieme (quindi i client messi assieme); la standardizzazione avviene successivamente con
            # le statistiche locali in data.py)
        nan_threshold=1.0, # tengo le serie anche con molti giorni mancanti perchè sono dispositivi accesi raramente e questo aiuta con la distinzione
            # tra end-device e server; in ogni caso i buchi vengono riempiti con il valore di default della feature
        random_state=cfg.RANDOM_STATE,
    )
    dataset.set_dataset_config_and_initialize(
        series_config, display_config_details=None, workers=workers
    )

    # inizio a caricare in memoria tutto il set come array numpy con forma (indirizzi, giorni, features)
    X = dataset.get_train_numpy(workers=workers)  # (n, T, F)
    got_ids = np.asarray(dataset.get_data_about_set("train")["ts_ids"])

    # devo effettuare un transpose perchè PyTorch vuole che le features siano il secondo argomento (indirizzi, features, giorni)
    X = np.transpose(X, (0, 2, 1)).astype(np.float32, copy=False)
    return np.ascontiguousarray(X), got_ids

def write_shard(path: Path, X: np.ndarray, y: np.ndarray, ts_ids: np.ndarray) -> None:
    np.savez(
        path,
        X=X.astype(np.float32, copy=False),
        y=y.astype(np.int64, copy=False),
        ts_ids=np.asarray(ts_ids),
    )


# pulisco la cartella prima di rigenerare gli shard
# non cancello lo shard del server
def clean_shard_dir() -> None:
    vecchi = sorted(cfg.SHARD_ROOT.glob("client_*.npz"))
    for percorso in vecchi:
        percorso.unlink()
    # anche i metadati, sennò descriverebbero una partizione che non esiste più
    if cfg.meta_path().exists():
        cfg.meta_path().unlink()
    if vecchi:
        log.info("Rimossi %d shard client della generazione precedente.", len(vecchi))


# test set del server valido sse gli indirizzi scelti dall'holdout (seed) sono corretti controllando gli identificatori
# (split diversi possono avere la stessa dimensione ma contenere cose diverse)
def server_test_valido(test_ids: np.ndarray) -> bool:
    percorso = cfg.server_shard_path()
    if not percorso.exists():
        return False
    try:
        with np.load(percorso) as data:
            presenti = np.asarray(data["ts_ids"])
    except Exception:
        # file troncato o illeggibile: meglio riscriverlo
        return False
    return len(presenti) == len(test_ids) and bool(
        np.array_equal(np.sort(presenti), np.sort(np.asarray(test_ids)))
    )


# Visualizzazione delle partizioni con informazioni sul contenuto (finisce dentro meta.json oltre che a essere mostrata sotto)
def summarize_partition(parts: list[np.ndarray], y: np.ndarray) -> dict:
    sizes = np.array([len(p) for p in parts])
    per_client = np.stack(
        [np.bincount(y[p], minlength=cfg.NUM_CLASSES) for p in parts]
    )

    return {
        "num_clients": len(parts),
        "samples_total": int(sizes.sum()),
        "samples_min": int(sizes.min()),
        "samples_max": int(sizes.max()),
        "samples_mean": float(sizes.mean()),
        "samples_median": float(np.median(sizes)),
        "clients_under_10": int((sizes < 10).sum()),
        "clients_under_20": int((sizes < 20).sum()),
        "class_counts_global": {
            name: int(per_client[:, idx].sum())
            for idx, name in enumerate(cfg.CLASS_NAMES)
        },
        "clients_without_class": {
            name: int((per_client[:, idx] == 0).sum())
            for idx, name in enumerate(cfg.CLASS_NAMES)
        },
        "per_client_class_counts": per_client.tolist(),
    }

# descrizione in chiaro di ogni modalità, usata sia nel riepilogo stampato sia in meta.json
DESCRIZIONE_PARTIZIONE = {
    "subnet": "una subnet istituzionale per client",
    "random": "indirizzi mescolati a caso, client di dimensione uguale",
    "random-sizes": "indirizzi mescolati a caso, dimensioni delle subnet vere",
}


def print_summary(summary: dict, modalita: str) -> None:
    mb = summary["samples_max"] * cfg.SEQ_LEN * cfg.NUM_FEATURES * 4 / 1024**2
    n = summary["num_clients"]
    print(f"\nPartizione fra i client ({DESCRIZIONE_PARTIZIONE[modalita]})")
    print(f"  client                     : {n}")
    print(f"  indirizzi distribuiti      : {summary['samples_total']:,}")
    print(
        f"  per client (min/mediana/max): {summary['samples_min']} / "
        f"{summary['samples_median']:.0f} / {summary['samples_max']}"
        f"   (media {summary['samples_mean']:.0f})"
    )
    print(
        f"  client sotto le 10 serie   : {summary['clients_under_10']}"
        f"   sotto le 20: {summary['clients_under_20']}"
    )
    print(f"  shard più grande           : {mb:.0f} MB in float32")
    print("  distribuzione delle classi :")
    for name in cfg.CLASS_NAMES:
        total = summary["class_counts_global"][name]
        missing = summary["clients_without_class"][name]
        print(
            f"    {name:<12} {total:>7,}   client che non ne hanno nemmeno uno: {missing}"
        )

    # la riga da mettere in pyproject.toml per supernodes
    print(
        f"\n  Metti questa riga in pyproject.toml, sezione "
        f"[tool.flwr.federations.local-simulation]:\n"
        f"      options.num-supernodes = {n}"
    )


# quando un errore ferma il download per un qualsiasi motivo, Ts-Zoo usa una funzione che può riprendere da dove aveva smesso di andare guardando
# quanti byte ci sono già sul disco e inviando una richiesta al server con header Range
def ensure_dataset_complete(data_root: Path) -> None:
    """Riprende il download se sul disco c'è un HDF5 troncato."""

    nome = "CESNET-TimeSeries24-ip_addresses_full-day"
    percorso = data_root / "tszoo" / "databases" / "CESNET-TimeSeries24" / f"{nome}.h5"
    if not percorso.exists():
        return  # Non c'è niente da riprendere: ci penserà `get_dataset`.

    import tables

    try:
        with tables.open_file(str(percorso), mode="r"):
            return  # Si apre, quindi è completo.
    # NOTA: per la libreria, un file incompleto è scartato e lancia normalmente un'eccezione
    except tables.exceptions.HDF5ExtError:
        pass

    mancanti = percorso.stat().st_size / 1024**2
    log.warning(
        "Il dataset in %s è incompleto (%.0f MB scaricati). Riprendo il download da dove si era interrotto.",
        percorso.name,
        mancanti,
    )
    # _download è un metodo interno della libreria e bisogna controllare che esista per il dataset (potrebbe essere rimosso in futuro)
    if not hasattr(CESNET_TimeSeries24, "_download"):
        raise SystemExit(
            f"Il file {percorso} è troncato e questa versione di cesnet-tszoo non "
            "espone più _download per riprenderlo. Cancellalo e rilancia lo "
            "script per riscaricarlo da capo."
        )
    CESNET_TimeSeries24._download(nome, str(percorso))


# apro il dataset (al primo giro si scarica) -> le etichette vengono lette -> viene effettuato lo split (come quello del centralizzato) ->
# -> faccio la divisione per subnet -> scrivo gli shard a blocchi e i metadati

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg.DATA_ROOT.mkdir(parents=True, exist_ok=True)
    cfg.SHARD_ROOT.mkdir(parents=True, exist_ok=True)

    log.info("Dataset in %s", cfg.DATA_ROOT)
    log.info("Shard in    %s", cfg.SHARD_ROOT)
    log.info("Divisione   %s (%s)", args.partition, DESCRIZIONE_PARTIZIONE[args.partition])
    if args.rebalance_net_device > 0:
        log.info("Ribilanciamento net-device a %d per client", args.rebalance_net_device)

    # shard precedenti cancellati mentre test set server resta
    clean_shard_dir()

    ensure_dataset_complete(cfg.DATA_ROOT)

    dataset = CESNET_TimeSeries24.get_dataset(
        data_root=str(cfg.DATA_ROOT),
        source_type=SourceType.IP_ADDRESSES_FULL,
        aggregation=AgreggationType.AGG_1_DAY,
        dataset_type=DatasetType.SERIES_BASED,
        display_details=False,
    )
    ts_id_col = dataset.metadata.ts_id_name

    labels = load_labels(dataset)

    # split
    ts_ids_all = labels.index.to_numpy()
    y_all = labels.to_numpy()

    pool_ids, pool_y_str, test_ids, test_y_str = holdout_split(
        ts_ids_all,
        y_all,
        random_state=cfg.RANDOM_STATE,
        test_ratio=cfg.TEST_RATIO,
        val_ratio=cfg.VAL_RATIO,
    )
    pool_y = encode_labels(pool_y_str)
    test_y = encode_labels(test_y_str)

    log.info(
        "Pool federato: %s indirizzi | test del server: %s indirizzi",
        f"{len(pool_ids):,}",
        f"{len(test_ids):,}",
    )

    # partizione per subnet istituzionale
    relationship = dataset.get_additional_data("ids_relationship")
    group_col = resolve_group_column(relationship, ts_id_col)
    log.info(
        "Partizione per gruppo naturale sulla colonna %r (%s gruppi nel dataset)",
        group_col,
        f"{relationship[group_col].nunique():,}",
    )

    group_of_ts = relationship.set_index(ts_id_col)[group_col]
    groups = group_of_ts.reindex(pool_ids)
    if groups.isna().any():
        n_missing = int(groups.isna().sum())
        raise SystemExit(
            f"{n_missing} indirizzi del pool non hanno un valore in {group_col!r}. "
        )
    parts = partition_by_group(groups.to_numpy())
    log.info("Subnet con almeno un indirizzo etichettato: %d", len(parts))
    parts = apply_min_subnet_size(parts, args.min_subnet_size)

    # la divisione per subnet viene semre calcolata perchè le latre divisioni si basano su di essa (num client e num indirizzi)
    if args.partition != "subnet":
        # tengo solo gli indirizzi che sono sopravvissuti alla soglia, cioè gli stessi 82.527
        indirizzi_del_pool = np.sort(np.concatenate(parts))
        dimensioni = [len(p) for p in parts] if args.partition == "random-sizes" else None

        posizioni = partition_random(
            len(indirizzi_del_pool),
            len(parts),
            seed=cfg.RANDOM_STATE,
            dimensioni=dimensioni,
        )
        # partition_random lavora su 0..n-1, quindi rimappo sulle posizioni vere dentro pool_ids
        parts = [indirizzi_del_pool[p] for p in posizioni]
        log.info(
            "Divisione %s: %d client sugli stessi %s indirizzi della partizione per subnet",
            args.partition,
            len(parts),
            f"{len(indirizzi_del_pool):,}",
        )

    summary = summarize_partition(parts, pool_y)
    print_summary(summary, args.partition)

    # prestiti[c] sono indici dell'insieme di indirizzi che il client c riceve da altri
    if args.rebalance_net_device > 0:
        prestiti = rebalance_net_device(
            parts, pool_y, args.rebalance_net_device, seed=cfg.RANDOM_STATE
        )
    else:
        prestiti = [np.array([], dtype=int) for _ in parts]

    # righe_di[c] contiene prima gli indirizzi del client c poi le serie ricevute in prestito
    righe_di = [
        np.concatenate([parts[c], prestiti[c]]) if len(prestiti[c]) else parts[c]
        for c in range(len(parts))
    ]

    # scrittura degli shard a blocchi di client 
    t0 = time.time()
    batch: list[int] = []
    batch_size = 0
    written = 0

    def flush(batch: list[int]) -> int:
        if not batch:
            return 0
        # np.unique perchè col ribilanciamento la stessa serie può servire a due client dello stesso blocco
        wanted = np.unique(np.concatenate([pool_ids[righe_di[c]] for c in batch]))
        X, got_ids = load_raw_series(dataset, wanted, workers=WORKERS)

        # riallineo per identificatore
        # position[ts_id] dice in quale riga di X sta quella serie.
        position = {int(tid): i for i, tid in enumerate(got_ids)}
        missing = [int(t) for t in wanted if int(t) not in position]
        if missing:
            raise RuntimeError(
                f"TS-Zoo non ha restituito {len(missing)} serie fra quelle "
                f"richieste (prime: {missing[:5]}). Controlla nan_threshold."
            )

        for client_id in batch:
            idx = righe_di[client_id]
            rows = np.array([position[int(t)] for t in pool_ids[idx]])
            write_shard(
                cfg.client_shard_path(client_id),
                X[rows],
                pool_y[idx],
                pool_ids[idx],
            )
        del X
        return len(batch)

    for client_id in range(len(parts)):
        n = len(righe_di[client_id])
        if batch and batch_size + n > CHUNK_SERIES:
            written += flush(batch)
            log.info("Shard scritti: %d/%d", written, len(parts))
            batch, batch_size = [], 0
        batch.append(client_id)
        batch_size += n
    written += flush(batch)
    log.info("Shard scritti: %d/%d", written, len(parts))

    # valido se test set del server è ancora valido altrimenti lo riscrivo
    if server_test_valido(test_ids):
        log.info(
            "Test set del server già presente e con gli stessi %s indirizzi.",
            f"{len(test_ids):,}",
        )
    else:
        log.info("Scrittura del test set del server (%s indirizzi)", f"{len(test_ids):,}")
        X_test, got_ids = load_raw_series(dataset, test_ids, workers=WORKERS)
        position = {int(t): i for i, t in enumerate(got_ids)}
        rows = np.array([position[int(t)] for t in test_ids])
        write_shard(cfg.server_shard_path(), X_test[rows], test_y, test_ids)
        del X_test


    # scrittura dei metadati
    # ServerApp e ClientApp li leggono per sapere quanti shard esistono e com'è fatta la partizione, altrimenti il controllo con il numero di SuperNodes non potrebbe essere effettuato
    meta = {
        "num_partitions": len(parts),
        # come sono stati divisi gli indirizzi; letto da ServerApp e grafici e finisce dentro file history
        "partition": args.partition,
        # la colonna delle subnet viene registrata a prescindere visto che anche nelle altre modalità il numero di client e colonna subnet derivano da qua
        "group_column": group_col,
        "min_subnet_size": args.min_subnet_size,
        # quante copie di net-device ha ricevuto ogni client da cui ricavo quante righe dello shard sono davvero sue
        "rebalance_net_device": int(args.rebalance_net_device),
        "own_per_client": [int(len(p)) for p in parts],
        "donated_per_client": [int(len(p)) for p in prestiti],
        "random_state": cfg.RANDOM_STATE,
        "class_names": list(cfg.CLASS_NAMES),
        "feature_names": list(cfg.FEATURE_NAMES),
        "seq_len": int(cfg.SEQ_LEN),
        "num_features": int(cfg.NUM_FEATURES),
        "pool_size": int(summary["samples_total"]), # indirizzi rimasti dopo aver scartato le subnet sotto soglia
        "pool_size_before_threshold": int(len(pool_ids)), # e quanti erano prima di scartarle
        "server_test_size": int(len(test_ids)),
        "server_test_class_counts": {
            name: int((test_y == idx).sum())
            for idx, name in enumerate(cfg.CLASS_NAMES)
        },
        "partition_summary": summary,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    cfg.meta_path().write_text(json.dumps(meta, indent=2), encoding="utf-8")

    elapsed = time.time() - t0
    print(
        f"\nFatto in {elapsed/60:.1f} min. "
        f"{len(parts)} shard client più il test del server in {cfg.SHARD_ROOT}"
    )
    print(f"Divisione: {args.partition} ({DESCRIZIONE_PARTIZIONE[args.partition]})")
    if args.rebalance_net_device > 0:
        print(
            f"Ribilanciamento: {sum(len(p) for p in prestiti):,} net-device copiati "
            f"per portare ogni client ad almeno {args.rebalance_net_device}"
        )
    print(
        "Ricordati che options.num-supernodes in pyproject.toml deve valere "
        f"{len(parts)}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
