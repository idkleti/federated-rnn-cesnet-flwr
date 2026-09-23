#!/usr/bin/env python3

# costruzioni grafici a partire da tutti gli esperimenti fatti

# per avviare:
#       python tools/grafici.py

# legge outputs/history_*.json e shards/meta.json e scrive in outputs/figure/

from __future__ import annotations

import json
import sys
from collections import defaultdict

from matplotlib.patches import Patch
from pathlib import Path

import matplotlib

# backend non interattivo perchè sono su wsl quindi niente plt.show()
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# voglio che quando viene lanciato venga messa in cima alla ricerca delle cartelle la radice del progetto
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedrnn import config as cfg 

# macro f1-score del test set del modello centralizzato con cui voglio confrontarmi 
BASELINE_MACRO_F1 = 0.7355 # appare in due figure come riferimento

COLORI = {
    "end-device": "#4C72B0",
    "net-device": "#C44E52",
    "server": "#55A868",
}

# ---------------------------------------------------------------------------
# Aspetto delle figure
#
# Modifica solo questo blocco per adattare le figure al template del paper.
# Le dimensioni sono in pollici (width, height) e i font in punti tipografici.
# Per una figura a due colonne, molti template usano una larghezza di 6.5--7 in;
# per una colonna, usa circa 3.3--3.5 in e riduci le dimensioni dei font insieme.
PLOT_CONFIG = {
    "dpi": 300,
    "base_font_size": 12,
    "tick_font_size": 11,
    "label_font_size": 13,
    "title_font_size": 15,
    "legend_font_size": 10.5,
    "annotation_font_size": 10.5,
    "curve_figsize": (10.5, 6.0),
    "curve_legend_columns": 2,
    "curve_legend_y": -0.25,
    "curve_bottom_margin": 0.34,
    "comparison_width": 10.0,
    "comparison_row_height": 0.76,
    "comparison_base_height": 2.6,
    "time_width": 10.0,
    "time_row_height": 0.62,
    "time_base_height": 2.4,
    "local_loss_figsize": (10.5, 6.0),
    "heatmap_panel_width": 5.4,
    "heatmap_panel_height": 3.8,
    "grid_alpha": 0.30,
    "iid_color": "#E17C05",
    "non_iid_color": "#4C72B0",
}

plt.rcParams.update({
    "font.size": PLOT_CONFIG["base_font_size"],
    "axes.labelsize": PLOT_CONFIG["label_font_size"],
    "axes.titlesize": PLOT_CONFIG["title_font_size"],
    "xtick.labelsize": PLOT_CONFIG["tick_font_size"],
    "ytick.labelsize": PLOT_CONFIG["tick_font_size"],
    "legend.fontsize": PLOT_CONFIG["legend_font_size"],
})


# lettura delle history degli esperimenti

# - all'interno di ogni file history_*.json ogni run scrive la propria configurazione e metriche round per round e si vanno a leggere SOLO le run complete
# - le run vengono raggruppate per partizione, ribilanciamento, strategia, fraction-train, decadimento lr, numero di round e passo del server

ROUND_MINIMI = 10

def carica_storici(cartella: Path) -> dict[tuple, list[dict]]:
    gruppi: dict[tuple, list[dict]] = defaultdict(list)
    for percorso in sorted(cartella.glob("history_*.json")):
        h = json.loads(percorso.read_text(encoding="utf-8"))
        run = h["run"]
        part = h.get("partition", {})
        n_round = int(run["num_rounds"])

        if n_round < ROUND_MINIMI:
            continue
        if not run.get("completed", False):
            continue
        # Dal protocollo corrente il test viene valutato una sola volta, sul
        # checkpoint selezionato dalla validation. Gli storici precedenti, con il
        # test disponibile a ogni round, non vengono mescolati a questi risultati.
        if run.get("test_evaluation") != "selected_checkpoint_once":
            continue
        if not h.get("selected_test_metrics"):
            continue
        if len(h.get("federated_eval_metrics", {})) < n_round:
            continue

        # il passo del server di FedAdam e il richiamo di FedProx fanno parte della chiave, altrimenti run con eta o con mu diverso verrebbero messe assieme.
        # ciascuno vale None per le strategie che non lo usano

        # partizione, ribilanciamento e mini-dataset vengono inseriti nella chiave per dividere run con stessi iperparametri ma dati distribuiti in modo diverso
        chiave = (
            run.get("strategy", "fedavg"),
            part["kind"],
            int(part.get("rebalance_net_device", 0) or 0),
            int(part.get("mini_dataset", 0) or 0),
            float(run["fraction_train"]),
            float(run.get("lr_decay", 1.0)),
            n_round,
            run.get("server_learning_rate"),
            run.get("proximal_mu"),
            run.get("server_momentum"),
            run.get("server_tau"),
        )
        gruppi[chiave].append(h)
    return gruppi


# come si chiama una partizione nelle legende dei grafici
PARTITION_LABELS = {
    "subnet": "non-IID",
    "random": "IID",
    "random-sizes": "IID (sizes)",
}

STRATEGY_LABELS = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "fedadam": "FedAdam",
    "fedyogi": "FedYogi",
    "fedavgm": "FedAvgM",
    "fedadagrad": "FedAdagrad",
    "fednova": "FedNova",
    "classaware": "CA-FedAvg",
    "class-aware": "CA-FedAvg",
}


def is_iid(chiave: tuple) -> bool:
    """Le due partizioni randomizzate sono le baseline IID."""
    return chiave[1] in {"random", "random-sizes"}


def selected_validation_macro_f1(history: dict) -> float:
    """Validation score used to select the checkpoint, never a test score."""
    return float(history["run"]["selected_fed_macro_f1"])


def iid_reference_configuration(groups: dict[tuple, list[dict]]) -> tuple | None:
    """Best randomized-IID configuration according to validation only."""
    candidates = [
        key for key in groups
        if key[1] == "random" and key[2] == 0 and key[3] == 0
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda key: float(np.mean([
            selected_validation_macro_f1(history) for history in groups[key]
        ])),
    )


def etichetta(chiave: tuple) -> str:
    (
        strategia,
        partizione,
        ribilanciamento,
        mini_dataset,
        ft,
        decay,
        n_round,
        server_lr,
        mu,
        server_momentum,
        server_tau,
    ) = chiave
    parti = [
        STRATEGY_LABELS.get(strategia, strategia),
        PARTITION_LABELS.get(partizione, partizione),
        f"f={ft:g}",
    ]
    if ribilanciamento:
        parti.append(f"RB={ribilanciamento}")
    if mini_dataset:
        parti.append(f"shared={mini_dataset}")
    # compare solo per FedAdam
    if server_lr is not None and server_lr != 0.01:
        parti.append(f"eta={server_lr:g}")
    # compare solo per FedProx
    if mu is not None:
        parti.append(f"mu={mu:g}")
    if strategia == "fedavgm" and server_momentum is not None:
        parti.append(f"m={server_momentum:g}")
    # Il numero di round e lr-decay restano nello storico JSON, ma non nella
    # label: rendono la legenda lunga senza distinguere la baseline scelta.
    return " · ".join(parti)


def serie_macro_f1(h: dict) -> tuple[np.ndarray, np.ndarray]:
    # La curva e' di validation federata: il test non viene usato round-per-round.
    c = h["federated_eval_metrics"]
    round_ = np.array(sorted(int(r) for r in c))
    valori = np.array([c[str(r)]["macro_f1"] for r in round_])
    return round_, valori


def macro_f1_selezionata(h: dict) -> float:
    # Macro-F1 del test, valutata una volta sul checkpoint gia' scelto con validation.
    return float(h["selected_test_metrics"]["macro_f1"])


def configurazione_consegnata(gruppi: dict[tuple, list[dict]]) -> tuple:
    # per sceglierla devo guardare la federazione vera (quella divisa per subnet e ribilanciamento)

    # di conseguenza le figure 3 e 4 mostrano solo cosa sbaglia il modello vero

    # si escludono le run in cui i client si sono prestati dati perchè non sono una federazione vera
    reali = [k for k in gruppi if k[1] == "subnet" and k[2] == 0 and k[3] == 0]
    if not reali:
        reali = list(gruppi)
    return max(
        reali,
        key=lambda k: float(np.mean([
            selected_validation_macro_f1(h) for h in gruppi[k]
        ])),
    )


def _salva(fig, percorso: Path) -> None:
    percorso.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(percorso, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)
    print("  wrote", percorso.name)


# FIGURA 1: CURVE DI APPRENDIMENTO
# ho una linea per configurazione della macro-f1 score per ogni round
# le curve grezze (linee leggermente trasparenti) oscillano troppo quindi disegno la media mobile sui 5 adiacenti con centro il round corrente
# sui bordi semplicemente diminuisco l'intervallo anzichè tagliare la curva

def media_mobile(valori: np.ndarray, finestra: int = 5) -> np.ndarray:
    fuori = np.empty_like(valori, dtype=float)
    meta = finestra // 2
    for i in range(len(valori)):
        fuori[i] = valori[max(0, i - meta) : i + meta + 1].mean()
    return fuori

def figura_curve(gruppi: dict[tuple, list[dict]], destinazione: Path) -> None:
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["curve_figsize"])
    iid_key = iid_reference_configuration(gruppi)

    # Draw the IID reference last so it remains visible when lines overlap.
    ordered_keys = [key for key in sorted(gruppi) if key != iid_key]
    if iid_key is not None:
        ordered_keys.append(iid_key)

    for key in ordered_keys:
        histories = gruppi[key]
        curves = np.stack([serie_macro_f1(history)[1] for history in histories])
        rounds = serie_macro_f1(histories[0])[0]
        mean_curve = curves.mean(axis=0)
        is_iid_reference = key == iid_key
        line, = ax.plot(
            rounds,
            media_mobile(mean_curve),
            linewidth=3.2 if is_iid_reference else 2.2,
            linestyle="--" if is_iid_reference else "-",
            color=PLOT_CONFIG["iid_color"] if is_iid_reference else None,
            zorder=4 if is_iid_reference else 2,
            label=etichetta(key),
        )
        ax.plot(
            rounds,
            mean_curve,
            linewidth=1.1 if is_iid_reference else 0.9,
            alpha=0.45 if is_iid_reference else 0.25,
            linestyle="--" if is_iid_reference else "-",
            color=line.get_color(),
            zorder=3 if is_iid_reference else 1,
        )

    ax.axhline(BASELINE_MACRO_F1, color="black", linestyle="--", linewidth=1.2)
    ax.text(1, BASELINE_MACRO_F1 + 0.012,
            f"Centralized RNN ({BASELINE_MACRO_F1:.4f})",
            fontsize=PLOT_CONFIG["annotation_font_size"])
    ax.set_xlabel("communication round")
    ax.set_ylabel("federated validation macro-F1")
    ax.set_title("Federated validation across communication rounds")
    ax.set_ylim(0, 0.85)
    ax.grid(alpha=PLOT_CONFIG["grid_alpha"])
    # Legenda fuori dall'area dei dati: non copre le curve a F1 elevata.
    ax.legend(
        fontsize=PLOT_CONFIG["legend_font_size"],
        loc="upper center",
        bbox_to_anchor=(0.5, PLOT_CONFIG["curve_legend_y"]),
        ncol=PLOT_CONFIG["curve_legend_columns"],
    )
    fig.subplots_adjust(bottom=PLOT_CONFIG["curve_bottom_margin"])
    _salva(fig, destinazione / "01_curve_macro_f1.png")


# FIGURA 2: CONFRONTO TRA STRATEGIE
# la barra è la macro-F1 sul test del server del modello consegnato mediata sulle due esecuzioni 
# il rombo invece è massimo ottenuto sul test durante la run, non selezionabile perché richiederebbe di guardare il test

def figura_confronto(gruppi: dict[tuple, list[dict]], destinazione: Path) -> None:
    # One bar per configuration. The test is evaluated once per replicate after
    # checkpoint selection on validation.
    entries = []
    for key in sorted(gruppi):
        values = np.asarray([macro_f1_selezionata(history) for history in gruppi[key]])
        entries.append((
            key,
            etichetta(key),
            float(values.mean()),
            float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            len(values),
        ))

    entries.sort(key=lambda entry: entry[2])
    labels = [entry[1] for entry in entries]
    y = np.arange(len(entries))
    # Tutte le partizioni randomizzate sono IID, non soltanto la migliore.
    colors = [
        PLOT_CONFIG["iid_color"] if is_iid(entry[0]) else PLOT_CONFIG["non_iid_color"]
        for entry in entries
    ]
    hatches = ["//" if is_iid(entry[0]) else None for entry in entries]

    fig, ax = plt.subplots(
        figsize=(
            PLOT_CONFIG["comparison_width"],
            PLOT_CONFIG["comparison_row_height"] * len(entries)
            + PLOT_CONFIG["comparison_base_height"],
        )
    )
    bars = ax.barh(y, [entry[2] for entry in entries], color=colors, edgecolor="black",
                   height=0.55, xerr=[entry[3] for entry in entries], capsize=3)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    for i, entry in enumerate(entries):
        ax.text(entry[2] - 0.012, i, f"{entry[2]:.4f}", va="center", ha="right",
                fontsize=PLOT_CONFIG["annotation_font_size"], color="white", fontweight="bold")

    ax.axvline(BASELINE_MACRO_F1, color="black", linestyle="--", linewidth=1.3)
    ax.text(BASELINE_MACRO_F1 - 0.01, len(entries) - 0.4,
            f"Centralized RNN\n{BASELINE_MACRO_F1:.4f}",
            fontsize=PLOT_CONFIG["annotation_font_size"], ha="right", va="top")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=PLOT_CONFIG["tick_font_size"])
    ax.set_xlabel("server test macro-F1")
    ax.set_title("Performance of validation-selected checkpoints")
    ax.set_xlim(0, 0.85)
    ax.set_ylim(-0.7, len(entries) - 0.3)
    ax.grid(alpha=PLOT_CONFIG["grid_alpha"], axis="x")
    legend_items = [
        Patch(facecolor=PLOT_CONFIG["non_iid_color"], edgecolor="black", label="non-IID"),
        Patch(facecolor=PLOT_CONFIG["iid_color"], edgecolor="black", hatch="//", label="IID"),
    ]
    ax.legend(handles=legend_items, fontsize=PLOT_CONFIG["legend_font_size"], loc="lower right")
    _salva(fig, destinazione / "02_confronto_strategie.png")


# FIGURA 7: TEMPI DI ADDESTRAMENTO
# Tempo end-to-end misurato dal ServerApp: statistiche federate, round e test
# finale del checkpoint scelto. Le barre includono la deviazione fra repliche.
def figura_tempi(gruppi: dict[tuple, list[dict]], destinazione: Path) -> None:
    entries = []
    for key in sorted(gruppi):
        seconds = np.asarray([
            float(history["run"]["elapsed_sec"])
            for history in gruppi[key]
            if history["run"].get("elapsed_sec") is not None
        ])
        if not len(seconds):
            continue
        entries.append((
            key,
            etichetta(key),
            float(seconds.mean() / 60),
            float(seconds.std(ddof=1) / 60) if len(seconds) > 1 else 0.0,
        ))

    if not entries:
        return
    entries.sort(key=lambda entry: entry[2])
    y = np.arange(len(entries))
    colors = [
        PLOT_CONFIG["iid_color"] if is_iid(entry[0]) else PLOT_CONFIG["non_iid_color"]
        for entry in entries
    ]
    fig, ax = plt.subplots(
        figsize=(
            PLOT_CONFIG["time_width"],
            PLOT_CONFIG["time_row_height"] * len(entries) + PLOT_CONFIG["time_base_height"],
        )
    )
    bars = ax.barh(
        y,
        [entry[2] for entry in entries],
        xerr=[entry[3] for entry in entries],
        color=colors,
        edgecolor="black",
        height=0.55,
        capsize=3,
    )
    for bar, entry in zip(bars, entries):
        ax.text(
            bar.get_width() + max(entry[3], 0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{entry[2]:.2f} min",
            va="center",
            fontsize=PLOT_CONFIG["annotation_font_size"],
        )
    ax.set_yticks(y)
    ax.set_yticklabels([entry[1] for entry in entries], fontsize=PLOT_CONFIG["tick_font_size"])
    ax.set_xlabel("end-to-end training time (minutes)")
    ax.set_title("Training time per configuration")
    ax.grid(alpha=PLOT_CONFIG["grid_alpha"], axis="x")
    ax.set_axisbelow(True)
    ax.set_xlim(left=0)
    _salva(fig, destinazione / "07_tempi_addestramento.png")


# FIGURE 8 e 9: DISPERSIONE DELLA LOSS LOCALE
# Le tracce client_train_metrics vengono scritte dalle nuove run. Non sono
# disponibili negli storici antecedenti a questa modifica.
def configurazioni_confronto_local_loss(
    gruppi: dict[tuple, list[dict]],
) -> list[tuple]:
    """Configurazioni non-IID comparabili: stessa pipeline, diversa strategia."""
    keys = []
    for key in sorted(gruppi):
        strategy, partition, rebalance, mini_dataset, fraction, decay, *_ = key
        if (
            partition != "subnet"
            or rebalance != 0
            or mini_dataset != 0
            or fraction != 0.5
            or decay != 0.97
        ):
            continue
        # Per FedProx la configurazione di riferimento resta quella di default.
        if strategy == "fedprox" and key[8] != 0.1:
            continue
        if any(history.get("client_train_metrics") for history in gruppi[key]):
            keys.append(key)
    return keys


def loss_locali_per_round(histories: list[dict]) -> dict[int, list[float]]:
    """Unisce le loss dei client campionati in tutte le repliche."""
    values: dict[int, list[float]] = defaultdict(list)
    for history in histories:
        for round_text, metrics in history.get("client_train_metrics", {}).items():
            for metric in metrics:
                loss = metric.get("train_loss")
                if loss is not None and np.isfinite(float(loss)):
                    values[int(round_text)].append(float(loss))
    return values


def figura_loss_locali_iqr(gruppi: dict[tuple, list[dict]], destinazione: Path) -> None:
    """Media e fascia 25--75% delle loss dei client partecipanti."""
    keys = configurazioni_confronto_local_loss(gruppi)
    if not keys:
        print("  no client-local loss traces: skipping local-loss figures")
        return

    fig, ax = plt.subplots(figsize=PLOT_CONFIG["local_loss_figsize"])
    for key in keys:
        per_round = loss_locali_per_round(gruppi[key])
        rounds = np.asarray(sorted(per_round))
        means = np.asarray([np.mean(per_round[round_]) for round_ in rounds])
        q25 = np.asarray([np.percentile(per_round[round_], 25) for round_ in rounds])
        q75 = np.asarray([np.percentile(per_round[round_], 75) for round_ in rounds])
        line, = ax.plot(rounds, means, linewidth=2.0, label=etichetta(key))
        ax.fill_between(rounds, q25, q75, color=line.get_color(), alpha=0.16)

    ax.set_xlabel("communication round")
    ax.set_ylabel("local training loss")
    ax.set_title("Mean local loss with interquartile band")
    ax.grid(alpha=PLOT_CONFIG["grid_alpha"])
    ax.legend(fontsize=PLOT_CONFIG["legend_font_size"], ncol=2)
    _salva(fig, destinazione / "08_loss_locali_iqr.png")


def figura_heatmap_loss_clienti(gruppi: dict[tuple, list[dict]], destinazione: Path) -> None:
    """Una heatmap per strategia, con righe ordinate per numerosita' client."""
    keys = configurazioni_confronto_local_loss(gruppi)
    if not keys:
        return

    all_losses = []
    matrices: list[tuple[tuple, np.ndarray, np.ndarray]] = []
    for key in keys:
        histories = gruppi[key]
        per_client_round: dict[tuple[int, int], list[float]] = defaultdict(list)
        max_round = 0
        max_client = -1
        for history in histories:
            for round_text, metrics in history.get("client_train_metrics", {}).items():
                round_ = int(round_text)
                max_round = max(max_round, round_)
                for metric in metrics:
                    if "client_id" not in metric or "train_loss" not in metric:
                        continue
                    client_id = int(metric["client_id"])
                    loss = float(metric["train_loss"])
                    if np.isfinite(loss):
                        per_client_round[(client_id, round_)].append(loss)
                        all_losses.append(loss)
                        max_client = max(max_client, client_id)

        summary = histories[0].get("partition", {}).get("summary", {})
        counts = np.asarray(summary.get("per_client_class_counts", []), dtype=float)
        client_sizes = counts.sum(axis=1) if counts.ndim == 2 else np.array([])
        n_clients = max(len(client_sizes), max_client + 1)
        if n_clients == 0 or max_round == 0:
            continue
        matrix = np.full((n_clients, max_round), np.nan)
        for (client_id, round_), losses in per_client_round.items():
            matrix[client_id, round_ - 1] = float(np.mean(losses))
        if len(client_sizes) == n_clients:
            order = np.argsort(-client_sizes)
        else:
            order = np.arange(n_clients)
        matrices.append((key, matrix[order], order))

    if not matrices or not all_losses:
        return
    vmin = 0.0
    vmax = max(float(np.percentile(all_losses, 95)), 1e-6)
    ncols = 2 if len(matrices) > 1 else 1
    nrows = int(np.ceil(len(matrices) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            ncols * PLOT_CONFIG["heatmap_panel_width"],
            nrows * PLOT_CONFIG["heatmap_panel_height"],
        ),
        squeeze=False,
    )
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#E6E6E6")  # client non campionato in quel round
    image = None
    for ax, (key, matrix, _) in zip(axes.flat, matrices):
        image = ax.imshow(
            np.ma.masked_invalid(matrix),
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        n_clients, n_rounds = matrix.shape
        ax.set_title(etichetta(key), fontsize=PLOT_CONFIG["annotation_font_size"])
        ax.set_xlabel("communication round")
        ax.set_ylabel("client rank by train samples")
        ax.set_xticks([0, max(0, n_rounds // 2 - 1), n_rounds - 1], [1, max(1, n_rounds // 2), n_rounds])
        ax.set_yticks([0, max(0, n_clients // 2 - 1), n_clients - 1], [1, max(1, n_clients // 2), n_clients])
    for ax in axes.flat[len(matrices):]:
        ax.set_visible(False)
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.flat[:len(matrices)], shrink=0.86)
        colorbar.set_label("mean local training loss")
    fig.suptitle("Client-local training loss (gray = not sampled)", fontsize=PLOT_CONFIG["title_font_size"])
    # `tight_layout` non gestisce il colorbar condiviso; margini espliciti
    # evitano sia sovrapposizioni sia il warning di Matplotlib.
    fig.subplots_adjust(left=0.14, right=0.90, bottom=0.08, top=0.91, wspace=0.34, hspace=0.45)
    _salva(fig, destinazione / "09_heatmap_loss_clienti.png")


# FIGURA 3: F1 PER CLASSE
# f1 delle tre classi round per round

def figura_per_classe(gruppi: dict[tuple, list[dict]], destinazione: Path) -> None:
    chiave = configurazione_consegnata(gruppi)
    run = gruppi[chiave]
    round_ = serie_macro_f1(run[0])[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    for nome in cfg.CLASS_NAMES:
        campo = "f1_" + nome.replace("-", "_")
        curve = np.stack([
            [h["federated_eval_metrics"][str(r)][campo] for r in round_]
            for h in run
        ])
        ax.plot(round_, curve.mean(axis=0), label=nome, linewidth=1.6,
                color=COLORI[nome])

    ax.set_xlabel("communication round")
    ax.set_ylabel("class F1 on federated validation")
    ax.set_title(f"Per-class F1 — {etichetta(chiave)}")
    ax.set_ylim(0, 1)
    ax.grid(alpha=PLOT_CONFIG["grid_alpha"])
    ax.legend(fontsize=PLOT_CONFIG["legend_font_size"])
    _salva(fig, destinazione / "03_f1_per_classe.png")


# FIGURA 4: MATRICE DI CONFUSIONE
# matrice di confusione del modello selezionato dalla configurazione migliore (nel mio caso FedAvg) sul test del server

def figura_confusione(gruppi: dict[tuple, list[dict]], destinazione: Path) -> None:
    chiave = configurazione_consegnata(gruppi)
    # Media delle matrici del checkpoint selezionato dalla validation, mai della
    # replica migliore sul test.
    matrices = np.stack([
        np.asarray(h["selected_test_metrics"]["confusion"], dtype=float)
        for h in gruppi[chiave]
    ])
    cm = matrices.mean(axis=0)
    perc = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    macro = float(np.mean([macro_f1_selezionata(h) for h in gruppi[chiave]]))

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.imshow(perc, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.1f}\n({perc[i, j]:.0%})",
                    ha="center", va="center", fontsize=PLOT_CONFIG["annotation_font_size"],
                    color="white" if perc[i, j] > 0.5 else "black")

    ax.set_xticks(range(cfg.NUM_CLASSES), cfg.CLASS_NAMES, rotation=20, ha="right")
    ax.set_yticks(range(cfg.NUM_CLASSES), cfg.CLASS_NAMES)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(f"Mean confusion matrix — {etichetta(chiave)}\n"
                 f"mean macro-F1 {macro:.4f}")
    _salva(fig, destinazione / "04_matrice_confusione.png")



# FIGURA 5: COME'E' FATTA LA FEDERAZIONE
# ho due pannelli:
#   - dimensione dei 69 client in scala logaritmica per rendere visibile il divario tra subnet grandi e subnet più piccole
#   - dove si trovano i net-device 

def figura_partizione(meta: dict, destinazione: Path) -> None:
    cc = np.array(meta["partition_summary"]["per_client_class_counts"])
    totali = cc.sum(axis=1)
    ordine = np.argsort(-totali)
    idx_net = cfg.CLASS_NAMES.index("net-device")

    fig, (sx, dx) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    sx.bar(range(len(totali)), totali[ordine], color="#4C72B0")
    sx.set_yscale("log")
    sx.set_xlabel("client rank, largest to smallest")
    sx.set_ylabel("IP addresses (log scale)")
    sx.set_title(f"Size of {len(totali)} clients")
    sx.grid(alpha=PLOT_CONFIG["grid_alpha"], axis="y")

    net = cc[ordine, idx_net]
    dx.bar(range(len(net)), net, color=COLORI["net-device"])
    piu_grande = int(np.argmax(net))
    dx.annotate(
        f"client {ordine[piu_grande]}: {int(net[piu_grande])} net-device\n"
        f"({100 * net[piu_grande] / max(net.sum(), 1):.0f}% of all net-device samples)",
        xy=(piu_grande, net[piu_grande]),
        xytext=(piu_grande + len(net) * 0.18, net[piu_grande] * 0.82),
        fontsize=PLOT_CONFIG["annotation_font_size"], arrowprops=dict(arrowstyle="->", lw=1),
    )
    dx.set_xlabel("client rank, same order as left panel")
    dx.set_ylabel("local net-device samples")
    dx.set_title("Location of the minority class")
    dx.grid(alpha=PLOT_CONFIG["grid_alpha"], axis="y")

    _salva(fig, destinazione / "05_partizione.png")


def figura_heatmap_label_skew(summary: dict, destinazione: Path) -> None:
    """Show each client's class proportions alongside its local sample count."""
    class_counts = np.asarray(summary["per_client_class_counts"], dtype=float)
    client_sizes = class_counts.sum(axis=1)
    order = np.argsort(-client_sizes)
    class_proportions = class_counts[order] / client_sizes[order, None]
    global_proportions = class_counts.sum(axis=0) / class_counts.sum()

    # Add the global distribution as a visual reference below all local clients.
    heatmap_data = np.vstack([class_proportions, global_proportions])
    n_clients = len(client_sizes)
    y = np.arange(n_clients)

    fig, (ax_size, ax_heatmap) = plt.subplots(
        1,
        2,
        figsize=(7.1, 7.2),
        gridspec_kw={"width_ratios": [1.35, 3.0], "wspace": 0.06},
        sharey=True,
    )

    ax_size.barh(y, client_sizes[order], color="#7F7F7F", height=0.82)
    ax_size.set_xscale("log")
    ax_size.invert_yaxis()
    ax_size.set_xlabel("samples\n(log scale)")
    ax_size.set_ylabel("client rank by sample count")
    ax_size.set_yticks([0, n_clients // 2, n_clients - 1])
    ax_size.set_yticklabels(["1", str(n_clients // 2 + 1), str(n_clients)])
    ax_size.grid(alpha=PLOT_CONFIG["grid_alpha"], axis="x")

    image = ax_heatmap.imshow(
        heatmap_data,
        aspect="auto",
        interpolation="nearest",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
    )
    ax_heatmap.axhline(n_clients - 0.5, color="black", linewidth=0.8)
    ax_heatmap.set_xticks(range(cfg.NUM_CLASSES), cfg.CLASS_NAMES, rotation=25, ha="right")
    ax_heatmap.set_yticks([0, n_clients // 2, n_clients - 1, n_clients])
    ax_heatmap.set_yticklabels(["1", str(n_clients // 2 + 1), str(n_clients), "global"])
    ax_heatmap.tick_params(axis="y", length=0)
    ax_heatmap.set_title("local class proportion")

    colorbar = fig.colorbar(image, ax=ax_heatmap, fraction=0.05, pad=0.03)
    colorbar.set_label("class proportion")
    fig.suptitle("Natural client partition: size and label composition", y=0.98)
    _salva(fig, destinazione / "06_label_skew_heatmap.png")


def main() -> int:
    gruppi = carica_storici(cfg.OUTPUT_ROOT)
    destinazione = cfg.OUTPUT_ROOT / "figure"

    if not gruppi:
        print(f"No protocol-compliant completed run in {cfg.OUTPUT_ROOT}. Run `flwr run .` first.")
        return 1

    print(f"Configurations found: {len(gruppi)}")
    for chiave in sorted(gruppi):
        print(f"  {etichetta(chiave)} — {len(gruppi[chiave])} runs")

    figura_curve(gruppi, destinazione)
    figura_confronto(gruppi, destinazione)
    figura_tempi(gruppi, destinazione)
    figura_loss_locali_iqr(gruppi, destinazione)
    figura_heatmap_loss_clienti(gruppi, destinazione)
    figura_per_classe(gruppi, destinazione)
    figura_confusione(gruppi, destinazione)

    # The selected configuration always supplies the partition statistics,
    # unlike shards/meta.json, which may not be present after cleanup.
    selected_key = configurazione_consegnata(gruppi)
    selected_history = max(gruppi[selected_key], key=selected_validation_macro_f1)
    summary = selected_history.get("partition", {}).get("summary")
    if summary:
        figura_heatmap_label_skew(summary, destinazione)
    else:
        print("  no partition summary: skipping label-skew heatmap")

    percorso_meta = cfg.meta_path()
    if percorso_meta.exists():
        figura_partizione(json.loads(percorso_meta.read_text(encoding="utf-8")),
                          destinazione)
    else:
        print(f"  {percorso_meta} is missing: skipping partition figure")

    print(f"\nFigures in {destinazione}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
