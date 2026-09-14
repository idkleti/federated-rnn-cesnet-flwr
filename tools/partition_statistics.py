#!/usr/bin/env python3
"""Report label-distribution heterogeneity for a federated partition.

The script reads the ``partition.summary`` stored in an experiment history and
computes the Jensen--Shannon divergence (base-2 logarithms) between each
client's label distribution and the global federated label distribution.

Examples:
    python tools/partition_statistics.py outputs/history_fedavg_subnet_ft0.5_lrd0.97_69c_50r_20260824-082045.json
    python tools/partition_statistics.py outputs/history_*.json --latex-summary
    python tools/partition_statistics.py outputs/history_*.json --latex-clients
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median


ROW_END = r"\\"


def kl_divergence(p: list[float], q: list[float]) -> float:
    """Return KL(p || q) in bits, ignoring zero-probability terms."""
    return sum(value * math.log2(value / reference) for value, reference in zip(p, q) if value)


def jensen_shannon_divergence(counts: list[int], global_distribution: list[float]) -> float:
    """Return JSD between one client's counts and the global distribution."""
    total = sum(counts)
    if total == 0:
        raise ValueError("A client has no samples.")
    local_distribution = [count / total for count in counts]
    mixture = [(local + global_) / 2 for local, global_ in zip(local_distribution, global_distribution)]
    return 0.5 * kl_divergence(local_distribution, mixture) + 0.5 * kl_divergence(global_distribution, mixture)


def load_partition(history_path: Path) -> tuple[list[str], list[int], list[list[int]]]:
    history = json.loads(history_path.read_text(encoding="utf-8"))
    try:
        summary = history["partition"]["summary"]
        class_counts = summary["class_counts_global"]
        per_client = summary["per_client_class_counts"]
    except KeyError as exc:
        raise ValueError(f"{history_path} does not contain partition statistics.") from exc

    class_names = list(class_counts)
    global_counts = [int(class_counts[name]) for name in class_names]
    return class_names, global_counts, [[int(value) for value in counts] for counts in per_client]


def latex_summary(values: list[float], sample_counts: list[int]) -> str:
    weighted_mean = sum(jsd * size for jsd, size in zip(values, sample_counts)) / sum(sample_counts)
    rows = [
        ("Mean across clients", sum(values) / len(values)),
        ("Median across clients", median(values)),
        ("Sample-weighted mean", weighted_mean),
        ("Minimum", min(values)),
        ("Maximum", max(values)),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Label-distribution heterogeneity across clients, measured by Jensen--Shannon divergence from the global federated label distribution. Values are expressed in bits.}",
        r"\label{tab:jsd-summary}",
        r"\small",
        r"\begin{tabular}{lr}",
        r"\toprule",
        "Statistic & JSD (bit) " + ROW_END,
        r"\midrule",
    ]
    lines.extend(f"{name} & {value:.6f} {ROW_END}" for name, value in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def latex_clients(values: list[float], sample_counts: list[int]) -> str:
    lines = [
        r"\begin{longtable}{rrr}",
        r"\caption{Per-client label-distribution heterogeneity. Clients are indexed in decreasing order of sample count.}",
        r"\label{tab:client-jsd}\\",
        r"\toprule",
        "Client ID & Samples & JSD (bit) " + ROW_END,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        "Client ID & Samples & JSD (bit) " + ROW_END,
        r"\midrule",
        r"\endhead",
    ]
    for client_id, (size, jsd) in enumerate(zip(sample_counts, values)):
        lines.append(f"{client_id} & {size:,} & {jsd:.6f} {ROW_END}")
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path, help="Path to one outputs/history_*.json file.")
    parser.add_argument("--latex-summary", action="store_true", help="Print a LaTeX summary table.")
    parser.add_argument("--latex-clients", action="store_true", help="Print a LaTeX per-client longtable.")
    args = parser.parse_args()

    class_names, global_counts, per_client_counts = load_partition(args.history)
    total = sum(global_counts)
    global_distribution = [count / total for count in global_counts]
    sample_counts = [sum(counts) for counts in per_client_counts]
    values = [jensen_shannon_divergence(counts, global_distribution) for counts in per_client_counts]

    print("Global label distribution:")
    for name, count, probability in zip(class_names, global_counts, global_distribution):
        print(f"  {name}: {count:,} ({probability:.2%})")
    print()

    if args.latex_clients:
        print(latex_clients(values, sample_counts))
    elif args.latex_summary:
        print(latex_summary(values, sample_counts))
    else:
        print("Client ID, samples, JSD (bit)")
        for client_id, (size, jsd) in enumerate(zip(sample_counts, values)):
            print(f"{client_id}, {size}, {jsd:.6f}")
        print()
        print(latex_summary(values, sample_counts))


if __name__ == "__main__":
    main()
