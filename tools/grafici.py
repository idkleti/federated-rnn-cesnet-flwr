#!/usr/bin/env python3

# costruzioni grafici a partire da tutti gli esperimenti fatti

# per avviare:
#       python tools/grafici.py

# leggere outputs/history_*.json e shards/meta.json e scrive in outputs/figure/

from __future__ import annotations

import json
import sys
from collections import defaultdict
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
BASELINE_MACRO_F1 = 0.7615 # appare in due figure come riferimento

COLORI = {
    "end-device": "#4C72B0",
    "net-device": "#C44E52",
    "server": "#55A868",
}


# lettura delle history degli esperimenti

# - all'interno di ogni file history_*.json ogni run scrive la propria configurazione e metriche round per round e si vanno a leggere 
#       SOLO le run complete
# - le run vengono raggruppate per strategia, fraction-train, decadimento lr, numero di round

ROUND_MINIMI = 10

def carica_storici(cartella: Path) -> dict[tuple, list[dict]]:
    gruppi: dict[tuple, list[dict]] = defaultdict(list)
    for percorso in sorted(cartella.glob("history_*.json")):
        h = json.loads(percorso.read_text(encoding="utf-8"))
        run = h["run"]
        n_round = int(run["num_rounds"])

        if n_round < ROUND_MINIMI:
            continue
        if h.get("partition", {}).get("kind") != "subnet":
            continue
        if not run.get("completed", False):
            continue
        if len(h["central_test_metrics"]) < n_round + 1:  # +1 per il round 0
            continue

        # il passo del server di FedAdam fa parte della chiave altrimenti run con eta diverso verrebbero messe assieme
        # per le altre strategie vale None
        chiave = (
            run.get("strategy", "fedavg"),
            float(run["fraction_train"]),
            float(run.get("lr_decay", 1.0)),
            n_round,
            run.get("server_learning_rate"),
        )
        gruppi[chiave].append(h)
    return gruppi


def etichetta(chiave: tuple) -> str:
    strategia, ft, decay, n_round, server_lr = chiave
    testo = f"{strategia}, partecipazione {ft:g}"
    if decay != 1.0:
        testo += f", lr ×{decay:g}"
    if n_round != 50:
        testo += f", {n_round} round"
    # compare solo per FedAdam
    if server_lr is not None and server_lr != 0.01:
        testo += f", eta {server_lr:g}"
    return testo


def serie_macro_f1(h: dict) -> tuple[np.ndarray, np.ndarray]:
    c = h["central_test_metrics"]
    round_ = np.array(sorted(int(r) for r in c))
    valori = np.array([c[str(r)]["macro_f1"] for r in round_])
    return round_, valori


def macro_f1_selezionata(h: dict) -> float:
    # macro-F1 sul test del modello che quella run ha effettivamente scelto
    return h["central_test_metrics"][str(h["run"]["selected_round"])]["macro_f1"]


def configurazione_migliore(gruppi: dict[tuple, list[dict]]) -> tuple:
    # la chiave viene scelta in base alla macro-f1 score selezionata più alta mediata sulle ripetizioni

    # calcolo la media perchè ho due esecuzioni per strategia
    return max( # restituisce la tupla con f1-score più alta
        gruppi,
        # key è il criterio con cui trovare il massimo
        key=lambda k: float(np.mean([macro_f1_selezionata(h) for h in gruppi[k]])),
    )


def _salva(fig, percorso: Path) -> None:
    percorso.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(percorso, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  scritto", percorso.name)


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
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for chiave in sorted(gruppi):
        run = gruppi[chiave]
        curve = np.stack([serie_macro_f1(h)[1] for h in run])
        round_ = serie_macro_f1(run[0])[0]
        media = curve.mean(axis=0) # media se config eseguita più volte
        # linea piena con media mobile
        linea, = ax.plot(round_, media_mobile(media), linewidth=2.2,
                         label=f"{etichetta(chiave)}  (n={len(run)})")
        # dato vero round per round
        ax.plot(round_, media, linewidth=0.9, alpha=0.25,
                color=linea.get_color())

    ax.axhline(BASELINE_MACRO_F1, color="black", linestyle="--", linewidth=1.2)
    ax.text(1, BASELINE_MACRO_F1 + 0.012,
            f"RNN centralizzata ({BASELINE_MACRO_F1:.4f})", fontsize=9)

    ax.set_xlabel("round")
    ax.set_ylabel("macro-F1 sul test del server")
    ax.set_title("Apprendimento federato round per round")
    ax.set_ylim(0, 0.85)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower right")
    _salva(fig, destinazione / "01_curve_macro_f1.png")


# FIGURA 2: CONFRONTO TRA STRATEGIE
# la barra è la macro-F1 sul test del server del modello consegnato mediata sulle due esecuzioni 
# il rombo invece è massimo ottenuto sul test durante la run, non selezionabile perché richiederebbe di guardare il test

def figura_confronto(gruppi: dict[tuple, list[dict]], destinazione: Path) -> None:
    voci = []
    for chiave in sorted(gruppi):
        sel, mas = [], []
        for h in gruppi[chiave]:
            c = h["central_test_metrics"]
            sel.append(c[str(h["run"]["selected_round"])]["macro_f1"])
            mas.append(max(m["macro_f1"] for m in c.values()))
        voci.append((etichetta(chiave), np.mean(sel), np.mean(mas), len(sel)))

    voci.sort(key=lambda v: v[1]) # ordina guardando il secondo elemento
    nomi = [v[0] for v in voci]
    y = np.arange(len(voci))

    fig, ax = plt.subplots(figsize=(9.5, 0.7 * len(voci) + 2.4))
    ax.barh(y, [v[1] for v in voci], color="#4C72B0", edgecolor="black",
            height=0.55, label="modello selezionato")

    for i, v in enumerate(voci):
        # posiziono il valore dentro la barra perchè sennò potrebbe scontrarsi con il rombo fuori
        ax.text(v[1] - 0.012, i, f"{v[1]:.4f}", va="center", ha="right",
                fontsize=9.5, color="white", fontweight="bold")
        ax.plot([v[2]], [i], marker="D", markersize=6, color="#C44E52",
                label="massimo raggiunto durante la run" if i == 0 else None)

    ax.axvline(BASELINE_MACRO_F1, color="black", linestyle="--", linewidth=1.3)
    # etichetta del riferimento va dentro il grafico in alto
    ax.text(BASELINE_MACRO_F1 - 0.01, len(voci) - 0.4,
            f"RNN centralizzata\n{BASELINE_MACRO_F1:.4f}",
            fontsize=9, ha="right", va="top")

    ax.set_yticks(y)
    ax.set_yticklabels(nomi, fontsize=9.5)
    ax.set_xlabel("macro-F1 sul test del server")
    ax.set_title("Quanto costa federare, per configurazione")
    ax.set_xlim(0, 0.85)
    ax.set_ylim(-0.7, len(voci) - 0.3)
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=8.5, loc="lower right")
    _salva(fig, destinazione / "02_confronto_strategie.png")


# FIGURA 3: F1 PER CLASSE
# f1 delle tre classi round per round

def figura_per_classe(gruppi: dict[tuple, list[dict]], destinazione: Path) -> None:
    chiave = configurazione_migliore(gruppi)
    # mostro quella che ha prodotto il modello migliore (nel mio caso FedAvg) fra le ripetizioni senza media (ne scelgo 1 sola)
    h = max(gruppi[chiave], key=macro_f1_selezionata)
    c = h["central_test_metrics"]
    round_ = np.array(sorted(int(r) for r in c))

    fig, ax = plt.subplots(figsize=(10, 5))
    for nome in cfg.CLASS_NAMES:
        campo = "f1_" + nome.replace("-", "_")
        ax.plot(round_, [c[str(r)][campo] for r in round_],
                label=nome, linewidth=1.6, color=COLORI[nome])

    ax.set_xlabel("round")
    ax.set_ylabel("F1 della classe sul test del server")
    ax.set_title(f"F1 per classe — {etichetta(chiave)}")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    _salva(fig, destinazione / "03_f1_per_classe.png")


# FIGURA 4: MATRICE DI CONFUSIONE
# matrice di confusione del modello selezionato dalla configurazione migliore (nel mio caso FedAvg) sul test del server

def figura_confusione(gruppi: dict[tuple, list[dict]], destinazione: Path) -> None:
    chiave = configurazione_migliore(gruppi)
    h = max(gruppi[chiave], key=macro_f1_selezionata)
    voce = h["central_test_metrics"][str(h["run"]["selected_round"])]
    cm = np.array(voce["confusion"], dtype=float)
    perc = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.imshow(perc, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(cm[i, j])}\n({perc[i, j]:.0%})",
                    ha="center", va="center", fontsize=10,
                    color="white" if perc[i, j] > 0.5 else "black")

    ax.set_xticks(range(cfg.NUM_CLASSES), cfg.CLASS_NAMES, rotation=20, ha="right")
    ax.set_yticks(range(cfg.NUM_CLASSES), cfg.CLASS_NAMES)
    ax.set_xlabel("predetto")
    ax.set_ylabel("reale")
    ax.set_title(f"Matrice di confusione — {etichetta(chiave)}\n"
                 f"macro-F1 {voce['macro_f1']:.4f}", fontsize=11)
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
    sx.set_xlabel("client, dal più grande al più piccolo")
    sx.set_ylabel("indirizzi (scala logaritmica)")
    sx.set_title(f"Dimensione dei {len(totali)} client")
    sx.grid(alpha=0.3, axis="y")

    net = cc[ordine, idx_net]
    dx.bar(range(len(net)), net, color=COLORI["net-device"])
    piu_grande = int(np.argmax(net))
    dx.annotate(
        f"client {ordine[piu_grande]}: {int(net[piu_grande])} net-device\n"
        f"({100 * net[piu_grande] / max(net.sum(), 1):.0f}% di tutta la classe)",
        xy=(piu_grande, net[piu_grande]),
        xytext=(piu_grande + len(net) * 0.18, net[piu_grande] * 0.82),
        fontsize=9, arrowprops=dict(arrowstyle="->", lw=1),
    )
    dx.set_xlabel("client, nello stesso ordine del pannello a sinistra")
    dx.set_ylabel("net-device posseduti")
    dx.set_title("Dove sta la classe minoritaria")
    dx.grid(alpha=0.3, axis="y")

    _salva(fig, destinazione / "05_partizione.png")


def main() -> int:
    gruppi = carica_storici(cfg.OUTPUT_ROOT)
    destinazione = cfg.OUTPUT_ROOT / "figure"

    if not gruppi:
        print(f"Nessuna run completa in {cfg.OUTPUT_ROOT}. Lancia prima `flwr run .`.")
        return 1

    print(f"Configurazioni trovate: {len(gruppi)}")
    for chiave in sorted(gruppi):
        print(f"  {etichetta(chiave)} — {len(gruppi[chiave])} esecuzioni")

    figura_curve(gruppi, destinazione)
    figura_confronto(gruppi, destinazione)
    figura_per_classe(gruppi, destinazione)
    figura_confusione(gruppi, destinazione)

    percorso_meta = cfg.meta_path()
    if percorso_meta.exists():
        figura_partizione(json.loads(percorso_meta.read_text(encoding="utf-8")),
                          destinazione)
    else:
        print(f"  {percorso_meta} non c'è: salto la figura della partizione")

    print(f"\nFigure in {destinazione}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
