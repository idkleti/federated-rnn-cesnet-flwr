# server app

# raccolgo statistiche dati -> costruisco modello iniziale -> far girare la strategia scelta per N round valutando a ogni round sul test set a parte ->
# -> salva modello e storico
from __future__ import annotations

import json
import logging
import time
from functools import partial

import numpy as np
import torch
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    MessageType,
    MetricRecord,
    RecordDict,
)

from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAdam, FedAvg, FedProx, FedYogi, Strategy

from fedrnn import config as cfg
from fedrnn.data import build_server_test_loader, load_meta, statistics_to_mean_std
from fedrnn.task import (
    DeviceRNN,
    balanced_class_weights,
    evaluate,
    format_confusion,
    metrics_from_confusion,
    pick_device,
    unflatten_confusion,
)

log = logging.getLogger("fedrnn.server")

app = ServerApp()

# chiavi con cui i client dichiarano quanti campioni hanno usato e con cui vengono pesate la media dei parametri e la media delle metriche scalari
WEIGHT_KEY = "num-examples"
CONFUSION_KEY = "confusion"

# partizioni
PARTIZIONI_AMMESSE = ("subnet", "random", "random-sizes")


# Collegamento di tutti i nodi

# le statistiche dovrebbero essere reperite da tutti i client, altrimenti la media e dev std sarebbero calcolate su una parte della federazione
# anzichè su tutta

# nel caso un timeout dovesse scadere la simulazione parte comunque 

def _timeout_per_nodi(expected: int, *, base: float, per_nodo: float) -> float:
    # parto da un'attesa fissa e ci aggiungo un tanto per ogni nodo che aspetto
    return base + per_nodo * expected

def wait_for_nodes(
    grid: Grid, expected: int, *, timeout: float | None = None
) -> list[int]:
    if timeout is None:
        timeout = _timeout_per_nodi(expected, base=120.0, per_nodo=1.0)
    deadline = time.time() + timeout
    node_ids: list[int] = []
    while time.time() < deadline:
        node_ids = list(grid.get_node_ids())
        if len(node_ids) >= expected:
            return node_ids
        time.sleep(1.0)
    log.warning(
        "Dopo %.0fs risultano collegati %d nodi su %d attesi. Si procede con quelli.",
        timeout,
        len(node_ids),
        expected,
    )
    return node_ids


# Scelta della strategia
# nonostante FedAvg sia il riferimento, è possibile tramite configurazione scegliere quale strategia utilizzare 

# FedAvg fa una media pesata degli aggiornamenti locali (niente memoria tra round e altro)
# FedAdam sostituisce la media semplice con un passo lato server tenendo le medie mobili degli aggiornamenti tra round
# FedProx utilizza la media di FedAvg e aggiunge alla perdita locale un termine che penalizza l'allontanarsi dai pesi con cui il round è iniziato
def build_strategy(
    name: str,
    *,
    fraction_train: float,
    fraction_evaluate: float,
    server_learning_rate: float,
    proximal_mu: float,
    train_sink: dict,
    evaluate_sink: dict,
) -> Strategy:
    # costruzione della strategia richiesta con aggregatori di metriche

    # non passo min_train_nodes/min_evaluate_nodes perchè flwr campiona max(frazione x nodi, minimo) e con 69 client la frazione dà sempre più del minimo (default 2)
    comuni = {
        "fraction_train": fraction_train,
        "fraction_evaluate": fraction_evaluate,
        "weighted_by_key": WEIGHT_KEY,
        "train_metrics_aggr_fn": partial(aggregate_train_metrics, sink=train_sink),
        "evaluate_metrics_aggr_fn": partial(
            aggregate_evaluate_metrics, sink=evaluate_sink
        ),
    }

    chiave = name.strip().lower()
    if chiave == "fedavg":
        return FedAvg(**comuni)
    if chiave == "fedprox":
        # proximal_mu è la forza del richiamo verso il modello globale.
        # se mu = 0 allora FedProx == FedAvg
        return FedProx(**comuni, proximal_mu=proximal_mu)
    if chiave == "fedadam":
        # eta è il passo del server
        return FedAdam(**comuni, eta=server_learning_rate)
    if chiave == "fedyogi":
        # come FedAdam, ma il passo del server non può crescere di colpo e non reagisce tanto alle run anomale
        return FedYogi(**comuni, eta=server_learning_rate)

    raise ValueError(
        f"Strategia {name!r} non riconosciuta. Valori ammessi: "
        "fedavg, fedprox, fedadam, fedyogi."
    )

# Round di statistiche
# il server, una volta ricevute le metriche, ricava media e varianza ottenute dalle somme aggregate e i pesi di classe (tutte queste metriche
# verranno inviate ai client tramite ConfigRecord successivamente)

def collect_federated_statistics(
    grid: Grid, num_expected: int, timeout: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # restituisce (media, deviazione standard, conteggi per classe)
    node_ids = wait_for_nodes(grid, num_expected)

    if timeout is None:
        # timeout elevato perchè bisogna leggere da disco lo shard e scorrerlo
        timeout = _timeout_per_nodi(len(node_ids), base=300.0, per_nodo=5.0)
    log.info(
        "Round di statistiche su %d nodi (timeout %.0f s)", len(node_ids), timeout
    )

    messages = [
        grid.create_message(
            content=RecordDict({"config": ConfigRecord({"task": "statistics"})}),
            message_type=MessageType.QUERY,
            dst_node_id=node_id,
            group_id="0",
        )
        for node_id in node_ids
    ]
    replies = grid.send_and_receive(messages, timeout=timeout)

    count = np.zeros(cfg.NUM_FEATURES, dtype=np.float64)
    total = np.zeros(cfg.NUM_FEATURES, dtype=np.float64)
    total_sq = np.zeros(cfg.NUM_FEATURES, dtype=np.float64)
    class_counts = np.zeros(cfg.NUM_CLASSES, dtype=np.float64)
    n_ok = 0

    for reply in replies:
        if reply.has_error():
            log.warning(
                "Nodo %d non ha risposto al round di statistiche: %s",
                reply.metadata.src_node_id,
                reply.error.reason,
            )
            continue
        metrics = reply.content["metrics"]
        count += np.asarray(metrics["stat_count"], dtype=np.float64)
        total += np.asarray(metrics["stat_sum"], dtype=np.float64)
        total_sq += np.asarray(metrics["stat_sum_sq"], dtype=np.float64)
        class_counts += np.asarray(metrics["class_count"], dtype=np.float64)
        n_ok += 1

    if n_ok == 0:
        raise RuntimeError(
            "Nessun client ha risposto al round di statistiche. Senza media e "
            "deviazione standard non si può standardizzare: controlla che gli "
            "shard esistano e che prepare_data.py sia stato eseguito."
        )

    mean, std = statistics_to_mean_std(count, total, total_sq)

    log.info(
        "Statistiche aggregate da %d client su %s valori per feature",
        n_ok,
        f"{int(count[0]):,}",
    )
    log.info(
        "Conteggi per classe nel pool federato: %s",
        {name: int(class_counts[i]) for i, name in enumerate(cfg.CLASS_NAMES)},
    )
    return mean, std, class_counts



# Come le risposte dei client diventano un numero solo
#
# i pesi del modello li aggrega Flower da sé mentre le metriche no: per quelle la
# libreria lascia due punti di innesto, train_metrics_aggr_fn e evaluate_metrics_aggr_fn
#
# il default farebbe la media pesata di ogni valore
# va bene per la loss ma non per la F1 che è un rapporto, e la media di rapporti non è il rapporto delle somme. 
# di conseguenza, vengono sommate le matrici di confusione dei client e calcolata la F1 una volta sola sul totale.

def _single_metric_record(record: RecordDict) -> dict | None:
    # estrae unico MetricRecord della risposta
    if not record.metric_records:
        return None
    key = next(iter(record.metric_records))
    return dict(record[key])


def aggregate_evaluate_metrics(
    records: list[RecordDict],
    weight_key: str = WEIGHT_KEY,
    *,
    sink: dict | None = None,
) -> MetricRecord:
    # somma delle matrici di confusione dei client e calcolo delle metriche globali con eventuale salvataggio parziale
    confusion = None
    total_weight = 0.0
    weighted_loss = 0.0
    server_round = 0
    n_clients = 0

    for record in records:
        metrics = _single_metric_record(record)
        if metrics is None or CONFUSION_KEY not in metrics:
            continue
        n_clients += 1

        weight = float(metrics.get(weight_key, 0.0))
        total_weight += weight
        weighted_loss += weight * float(metrics.get("eval_loss", 0.0))
        server_round = max(server_round, int(metrics.get("server-round", 0)))

        cm = unflatten_confusion(metrics[CONFUSION_KEY])
        confusion = cm if confusion is None else confusion + cm

    if confusion is None:
        return MetricRecord({weight_key: 0.0})

    out = metrics_from_confusion(confusion)
    out["eval_loss"] = weighted_loss / total_weight if total_weight else 0.0
    out[weight_key] = total_weight
    out["participating_clients"] = float(n_clients)

    log.info(
        "Round %d | valutazione federata su %d client, %d indirizzi | "
        "macro-F1 %.4f | accuratezza %.4f | loss %.4f",
        server_round,
        n_clients,
        int(total_weight),
        out["macro_f1"],
        out["accuracy"],
        out["eval_loss"],
    )

    if sink is not None:
        sink[str(server_round)] = {**out, "confusion": confusion.reshape(-1).tolist()}
    return MetricRecord(out)


def aggregate_train_metrics(
    records: list[RecordDict],
    weight_key: str = WEIGHT_KEY,
    *,
    sink: dict | None = None,
) -> MetricRecord:
    # media pesata della perdita di addestramento perchè durante il training i client riportano la loss (che è una media)
    total_weight = 0.0
    weighted_loss = 0.0
    total_batches = 0.0
    server_round = 0
    n_clients = 0

    for record in records:
        metrics = _single_metric_record(record)
        if metrics is None:
            continue
        n_clients += 1
        weight = float(metrics.get(weight_key, 0.0))
        total_weight += weight
        weighted_loss += weight * float(metrics.get("train_loss", 0.0))
        total_batches += float(metrics.get("num_batches", 0.0))
        server_round = max(server_round, int(metrics.get("server-round", 0)))

    out = {
        "train_loss": weighted_loss / total_weight if total_weight else 0.0,
        weight_key: total_weight,
        "num_batches": total_batches,
        "participating_clients": float(n_clients),
    }
    log.info(
        "Round %d | addestramento su %d client, %d indirizzi | loss %.4f",
        server_round,
        n_clients,
        int(total_weight),
        out["train_loss"],
    )
    if sink is not None:
        sink[str(server_round)] = dict(out)
    return MetricRecord(out)


# statistiche -> modello iniziale (costruito dal server e spedito identico ai client) -> strategia -> salvataggi
@app.main()
def main(grid: Grid, context: Context) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    run = context.run_config
    num_rounds = int(run["num-server-rounds"])
    fraction_train = float(run["fraction-train"])
    fraction_evaluate = float(run["fraction-evaluate"])
    lr = float(run["learning-rate"])
    lr_decay = float(run["lr-decay"])
    strategy_name = str(run["strategy"])
    server_lr = float(run["server-learning-rate"])
    proximal_mu = float(run["proximal-mu"])

    meta = load_meta()
    num_partitions = int(meta["num_partitions"])


    # controllo che la run venga effettuata sui dati veri e non finti (errore di scrittura di partizione)
    # le modalità casuali sono ammesse perchè usano gli stessi indirizzi veri mentre tutto il resto viene rifiutato
    partizione = str(meta.get("partition"))
    if partizione not in PARTIZIONI_AMMESSE:
        raise RuntimeError(
            f"Gli shard in {cfg.SHARD_ROOT} dichiarano partizione "
            f"{partizione!r}, che non è fra {PARTIZIONI_AMMESSE}. Questo progetto si "
            "addestra solo su dati veri: rigenerali con python prepare_data.py."
        )
    # quante copie di net-device sono state distribuite ai client che non ne avevano (0 = nessuna)
    ribilanciamento = int(meta.get("rebalance_net_device", 0) or 0)
    mini_dataset = int(meta.get("mini_dataset", 0) or 0)

    log.info(
        "Partizione per %s su %d client (colonna %s)",
        meta["partition"],
        num_partitions,
        meta.get("group_column"),
    )
    log.info(
        "Pool federato: %s indirizzi | test del server: %s indirizzi",
        f"{meta['pool_size']:,}",
        f"{meta['server_test_size']:,}",
    )
    
    if ribilanciamento:
        log.info(
            "RIBILANCIAMENTO ATTIVO: ogni client ha almeno %d net-device, %s copie "
            "distribuite fra i client (ipotesi di riservatezza rilassata)",
            ribilanciamento,
            f"{sum(meta.get('donated_per_client', [])):,}",
        )
    log.info("Strategia: %s | round: %d", strategy_name, num_rounds)

    
    # statistiche federate, una volta sola
    mean, std, class_counts = collect_federated_statistics(grid, num_partitions)
    class_weights = balanced_class_weights(class_counts)
    log.info(
        "Pesi di classe: %s",
        {
            name: round(float(class_weights[i]), 4)
            for i, name in enumerate(cfg.CLASS_NAMES)
        },
    )


    # se invio i vettori ogni volta anzichè una volta sola posso evitare di dover mantenere uno stato sul client fra un round e l'altro
    shared_config = {
        "lr": lr,
        "lr-decay": lr_decay,
        "feature-mean": [float(v) for v in mean],
        "feature-std": [float(v) for v in std],
        "class-weights": [float(v) for v in class_weights],
    }
    
    # modello iniziale
    global_model = DeviceRNN()
    initial_arrays = ArrayRecord(global_model.state_dict())

    device = pick_device()

    # dato che strategy.start salva le metriche solamente alla fine della run, per evitare che non vengano salvate causa problemi esterni,
    # uso dei dizionari per tenere traccia delle metriche round per round
    history_train: dict[str, dict] = {}
    history_fed_eval: dict[str, dict] = {}
    # Il test non entra nel ciclo di selezione: viene eseguito una sola volta,
    # dopo che la validation federata ha scelto il checkpoint.
    selected_test_metrics: dict[str, float | list] | None = None

    cfg.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    # il nome contiene tutti i dati per distinguere le run (strategia, frazione di partecipanti, decadimento, partizione, ribilanciamento, mini-dataset, mu di FedProx)
    tag = (
        f"{strategy_name}_{partizione}"
        f"{f'-rb{ribilanciamento}' if ribilanciamento else ''}"
        f"{f'-md{mini_dataset}' if mini_dataset else ''}"
        f"_ft{fraction_train:g}_lrd{lr_decay:g}"
        f"{f'_mu{proximal_mu:g}' if strategy_name == 'fedprox' else ''}"
        f"_{num_partitions}c_{num_rounds}r"
    )
    history_path = cfg.OUTPUT_ROOT / f"history_{tag}_{stamp}.json"
    t0 = time.time()

    
    # Modello consegnato
    # il modello finale migliore viene salvato in base alla macro f1 della valutazione federata (non sul test set perchè sennò non rappresenterebbe
    # una situazione reale)
    best = {"fed_macro_f1": -1.0, "round": 0}
    best_model_path = cfg.OUTPUT_ROOT / f"model_{tag}_{stamp}_best.pt"

    def salva_storico(completo: bool) -> None:
        history = {
            "run": {
                "strategy": strategy_name,
                "lr_decay": lr_decay,
                "server_learning_rate": (
                    server_lr if strategy_name in ("fedadam", "fedyogi") else None
                ),
                "proximal_mu": proximal_mu if strategy_name == "fedprox" else None,
                "num_rounds": num_rounds,
                "num_partitions": num_partitions,
                "fraction_train": fraction_train,
                "fraction_evaluate": fraction_evaluate,
                "local_epochs": int(run["local-epochs"]),
                "learning_rate": lr,
                "weight_decay": float(run["weight-decay"]),
                "batch_size": int(run["batch-size"]),
                "device": str(device),
                "elapsed_sec": time.time() - t0,
                "selected_round": best["round"],
                "selected_fed_macro_f1": best["fed_macro_f1"],
                "completed": completo, # chi legge lo storico deve essere in grado di capire se un esperimento è stato completato o meno
                "rounds_done": len(history_fed_eval),
                "test_evaluation": "selected_checkpoint_once",
                "test_evaluations": 1 if selected_test_metrics is not None else 0,
            },
            "partition": {
                "kind": meta["partition"],
                "group_column": meta.get("group_column"),
                "rebalance_net_device": ribilanciamento,
                "mini_dataset": mini_dataset,
                "summary": meta["partition_summary"],
            },
            "feature_mean": [float(v) for v in mean],
            "feature_std": [float(v) for v in std],
            "class_counts": [float(v) for v in class_counts],
            "class_weights": [float(v) for v in class_weights],
            "train_metrics": history_train,
            "federated_eval_metrics": history_fed_eval,
            # Campo separato: il test non e' una serie temporale per scegliere
            # round o configurazioni a posteriori.
            "selected_test_metrics": selected_test_metrics,
        }
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    log.info("Output della run: %s", cfg.OUTPUT_ROOT.resolve())
    salva_storico(completo=False)
    log.info("Storico iniziale salvato in %s", history_path)

    
    
    # Callback invocato dopo l'aggregazione della validation client-side. Seleziona
    # il checkpoint esclusivamente con quella validation: non carica ne' valuta il
    # test set.
    best_state_dict: dict[str, torch.Tensor] | None = None

    def select_checkpoint(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        nonlocal best_state_dict

        # Il round 0 non ha una validation federata.
        fed = history_fed_eval.get(str(server_round))
        if fed is not None and fed["macro_f1"] > best["fed_macro_f1"]:
            best["fed_macro_f1"] = float(fed["macro_f1"])
            best["round"] = server_round
            best_state_dict = {
                name: value.detach().cpu().clone()
                for name, value in arrays.to_torch_state_dict().items()
            }
            torch.save(best_state_dict, best_model_path)
            log.info(
                "   nuovo checkpoint al round %d (macro-F1 validation federata %.4f)",
                server_round,
                fed["macro_f1"],
            )

        # Salvataggio parziale, senza alcuna metrica di test.
        salva_storico(completo=False)
        return MetricRecord({"selected_validation_round": float(best["round"])})


    # Costruzione della strategia
    strategy = build_strategy(
        strategy_name,
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        server_learning_rate=server_lr,
        proximal_mu=proximal_mu,
        train_sink=history_train,
        evaluate_sink=history_fed_eval,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        train_config=ConfigRecord(shared_config),
        evaluate_config=ConfigRecord(shared_config),
        evaluate_fn=select_checkpoint,
    )
    elapsed = time.time() - t0

    if best_state_dict is None:
        raise RuntimeError(
            "Nessuna validation federata disponibile: il checkpoint non puo' "
            "essere scelto senza usare il test set."
        )

    # Unica valutazione sul test, eseguita solo sul checkpoint gia' selezionato.
    selected_model = DeviceRNN()
    selected_model.load_state_dict(best_state_dict)
    loader = build_server_test_loader(batch_size=256, mean=mean, std=std)
    loss, confusion, n_samples = evaluate(
        selected_model, loader, class_weights=class_weights, device=device
    )
    selected_test_metrics = {
        **{k: float(v) for k, v in metrics_from_confusion(confusion).items()},
        "test_loss": float(loss),
        WEIGHT_KEY: float(n_samples),
        "selected_round": float(best["round"]),
        "confusion": confusion.tolist(),
    }
    log.info(
        "TEST una sola volta sul checkpoint scelto dalla validation (round %d) | "
        "macro-F1 %.4f | F1 end-device %.4f, net-device %.4f, server %.4f\n%s",
        best["round"],
        selected_test_metrics["macro_f1"],
        selected_test_metrics["f1_end_device"],
        selected_test_metrics["f1_net_device"],
        selected_test_metrics["f1_server"],
        format_confusion(confusion),
    )

    # Salvataggi finali
    model_path = cfg.OUTPUT_ROOT / f"model_{tag}_{stamp}.pt"
    torch.save(result.arrays.to_torch_state_dict(), model_path)
    salva_storico(completo=True)
    log.info("Modello finale salvato in %s", model_path)
    log.info("Storico salvato in %s", history_path)

    log.info("=" * 78)
    log.info(
        "Fine. %d round in %.1f min (%.1fs per round)",
        num_rounds,
        elapsed / 60,
        elapsed / max(num_rounds, 1),
    )
    log.info(
        "macro-F1 sul test, checkpoint selezionato: %.4f  (round %d, scelto con "
        "macro-F1 validation federata %.4f)",
        selected_test_metrics["macro_f1"],
        best["round"],
        best["fed_macro_f1"],
    )
    log.info("   salvato in %s", best_model_path.name)
    log.info(
        "Riferimento: RNN centralizzata con pesi di classe, macro-F1 0.7615 "
        "sugli stessi 9.238 indirizzi"
    )
