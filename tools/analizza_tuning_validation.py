#!/usr/bin/env python3
"""Riassume esclusivamente le run di tuning senza test centrale.

Legge gli ``history_*.json`` e considera solo gli storici completati con
``test_evaluation == \"disabled\"``. La macro-F1 qui riportata e' quindi quella
della validation federata usata per scegliere il checkpoint, non una metrica di
test. Questo file non avvia ne' valuta esperimenti.
"""

from __future__ import annotations

import csv
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedrnn import config as cfg


STRATEGY_LABELS = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "fedyogi": "FedYogi",
    "classaware": "CA-FedAvg",
}


def _load_validation_only_histories(output_root: Path) -> list[dict]:
    histories: list[dict] = []
    for path in sorted(output_root.glob("history_*.json")):
        try:
            history = json.loads(path.read_text(encoding="utf-8"))
            run = history["run"]
            if not run.get("completed", False):
                continue
            if run.get("test_evaluation") != "disabled":
                continue
            if "selected_fed_macro_f1" not in run:
                continue
            history["_path"] = path.name
            histories.append(history)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            print(f"Skipping malformed history {path.name}: {exc}", file=sys.stderr)
    return histories


def _key(history: dict) -> tuple[str, float, int, float | None, int, float]:
    run = history["run"]
    return (
        str(run["strategy"]),
        float(run["fraction_train"]),
        int(run["local_epochs"]),
        float(run["proximal_mu"]) if run.get("proximal_mu") is not None else None,
        int(run["num_rounds"]),
        float(run["lr_decay"]),
    )


def _label(key: tuple[str, float, int, float | None, int, float]) -> str:
    strategy, fraction, epochs, mu, _rounds, _decay = key
    label = f"{STRATEGY_LABELS.get(strategy, strategy)} | f={fraction:g}, E={epochs}"
    if mu is not None:
        label += f", μ={mu:g}"
    return label


def _summarize(histories: list[dict]) -> list[dict]:
    groups: dict[tuple[str, float, int, float | None, int, float], list[dict]] = defaultdict(list)
    for history in histories:
        groups[_key(history)].append(history)

    rows: list[dict] = []
    for key, group in groups.items():
        scores = np.asarray([float(h["run"]["selected_fed_macro_f1"]) for h in group])
        selected_rounds = np.asarray([float(h["run"]["selected_round"]) for h in group])
        elapsed_minutes = np.asarray([float(h["run"].get("elapsed_sec", 0.0)) / 60 for h in group])
        strategy, fraction, epochs, mu, rounds, decay = key
        rows.append({
            "strategy": strategy,
            "fraction_train": fraction,
            "local_epochs": epochs,
            "proximal_mu": mu,
            "num_rounds": rounds,
            "lr_decay": decay,
            "n_runs": len(group),
            "validation_macro_f1_mean": float(scores.mean()),
            "validation_macro_f1_std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
            "selected_round_mean": float(selected_rounds.mean()),
            "elapsed_minutes_mean": float(elapsed_minutes.mean()),
            "label": _label(key),
        })
    return sorted(rows, key=lambda row: row["validation_macro_f1_mean"], reverse=True)


def _write_table(rows: list[dict], tables_dir: Path) -> Path:
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / "validation_tuning_summary.csv"
    fieldnames = [
        "strategy", "fraction_train", "local_epochs", "proximal_mu",
        "num_rounds", "lr_decay", "n_runs", "validation_macro_f1_mean",
        "validation_macro_f1_std", "selected_round_mean", "elapsed_minutes_mean",
        "label",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("  wrote", path.name)
    return path


def _load_existing_summary(path: Path) -> list[dict]:
    """Ricarica il riepilogo se gli history di tuning sono stati archiviati.

    La tabella e' generata dallo stesso script al termine del tuning e non
    contiene metriche di test; e' quindi un fallback sicuro per rigenerare la
    figura anche dopo una pulizia degli history intermedi.
    """
    numeric_fields = {
        "fraction_train", "local_epochs", "num_rounds", "lr_decay", "n_runs",
        "validation_macro_f1_mean", "validation_macro_f1_std",
        "selected_round_mean", "elapsed_minutes_mean",
    }
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as file:
        for raw in csv.DictReader(file):
            row = dict(raw)
            for name in numeric_fields:
                row[name] = float(row[name])
            row["local_epochs"] = int(row["local_epochs"])
            row["num_rounds"] = int(row["num_rounds"])
            row["n_runs"] = int(row["n_runs"])
            row["proximal_mu"] = (
                float(row["proximal_mu"]) if row.get("proximal_mu") else None
            )
            rows.append(row)
    return rows


def _plot(rows: list[dict], figure_dir: Path) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure, (ablation_ax, mu_ax) = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)

    # Le sei configurazioni dello sweep hanno f=.5 ed E=2; μ=.1 viene
    # dall'ablazione, perciò tutti i punti FedProx con quei fattori sono qui.
    mu_rows = [
        row for row in rows
        if row["strategy"] == "fedprox"
        and row["fraction_train"] == 0.5
        and row["local_epochs"] == 2
    ]
    mu_values = {row["proximal_mu"] for row in mu_rows}
    ablation_rows = [
        row for row in rows
        if not (
            row["strategy"] == "fedprox"
            and row["fraction_train"] == 0.5
            and row["local_epochs"] == 2
            and row["proximal_mu"] in mu_values
        )
    ]
    # Riaggiunge μ=.1 nel pannello di ablation: e' il FedProx standard del
    # disegno fattoriale; gli altri μ rimangono confinati allo sweep.
    ablation_rows.extend(row for row in mu_rows if row["proximal_mu"] == 0.1)
    ablation_rows.sort(key=lambda row: row["validation_macro_f1_mean"])

    labels = [row["label"] for row in ablation_rows]
    means = [row["validation_macro_f1_mean"] for row in ablation_rows]
    stds = [row["validation_macro_f1_std"] for row in ablation_rows]
    colors = ["#C44E52" if row["strategy"] == "fedprox" else "#4C72B0" for row in ablation_rows]
    ablation_ax.barh(range(len(ablation_rows)), means, xerr=stds, color=colors, capsize=3)
    ablation_ax.set_yticks(range(len(ablation_rows)), labels, fontsize=9)
    ablation_ax.set_xlabel("Selected federated validation macro-F1")
    ablation_ax.set_title("Validation-only ablation")
    ablation_ax.grid(axis="x", alpha=0.3)
    if means:
        ablation_ax.set_xlim(max(0, min(means) - 0.04), min(1, max(means) + 0.04))

    mu_rows.sort(key=lambda row: float(row["proximal_mu"]))
    if mu_rows:
        mus = np.asarray([float(row["proximal_mu"]) for row in mu_rows])
        means = np.asarray([row["validation_macro_f1_mean"] for row in mu_rows])
        stds = np.asarray([row["validation_macro_f1_std"] for row in mu_rows])
        mu_ax.errorbar(mus, means, yerr=stds, marker="o", linewidth=2.2, capsize=4, color="#C44E52")
        mu_ax.set_xscale("log")
        mu_ax.set_xticks(mus, [f"{mu:g}" for mu in mus])
        mu_ax.set_ylim(max(0, float(np.min(means - stds)) - 0.04), min(1, float(np.max(means + stds)) + 0.04))
    mu_ax.set_xlabel("FedProx μ (log scale)")
    mu_ax.set_ylabel("Selected federated validation macro-F1")
    mu_ax.set_title("FedProx μ sweep: f=0.5, E=2")
    mu_ax.grid(alpha=0.3)
    figure.suptitle("Hyperparameter selection uses validation only — no central test evaluation", fontsize=14)

    path = figure_dir / "15_tuning_validation_only.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print("  wrote", path.name)
    return path


def _refresh_archive(output_root: Path) -> Path:
    path = output_root / "figure_tables.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for folder_name in ("figure", "tables"):
            folder = output_root / folder_name
            if folder.exists():
                for item in sorted(folder.rglob("*")):
                    if item.is_file():
                        archive.write(item, item.relative_to(output_root))
    print("  wrote", path.name)
    return path


def main() -> int:
    histories = _load_validation_only_histories(cfg.OUTPUT_ROOT)
    if not histories:
        summary = cfg.OUTPUT_ROOT / "tables" / "validation_tuning_summary.csv"
        if not summary.exists():
            print("No completed validation-only tuning histories or summary table found.")
            return 1
        rows = _load_existing_summary(summary)
        print(
            "Validation-only histories are archived; regenerating the figure "
            f"from {summary.name} ({len(rows)} configurations)."
        )
    else:
        rows = _summarize(histories)
        print(f"Validation-only runs: {len(histories)} | configurations: {len(rows)}")
        _write_table(rows, cfg.OUTPUT_ROOT / "tables")
    _plot(rows, cfg.OUTPUT_ROOT / "figure")
    _refresh_archive(cfg.OUTPUT_ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
