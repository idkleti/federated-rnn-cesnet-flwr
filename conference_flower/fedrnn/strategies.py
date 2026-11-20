"""Strategie di aggregazione aggiuntive per l'API Message di Flower 1.32."""

from __future__ import annotations

import logging
from collections.abc import Iterable
import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

log = logging.getLogger("fedrnn.strategies")

# Istogramma delle classi effettivamente usate per l'addestramento locale.
TRAIN_CLASS_COUNTS_KEY = "train-class-count"
LOCAL_STEPS_KEY = "num_batches"


def _metric_record(record: RecordDict) -> MetricRecord:
    """Restituisce il solo MetricRecord, già validato da FedAvg."""
    return next(iter(record.metric_records.values()))


def _sample_weights(records: list[RecordDict], weight_key: str) -> np.ndarray:
    weights = np.asarray(
        [float(_metric_record(record)[weight_key]) for record in records],
        dtype=np.float64,
    )
    total = float(weights.sum())
    if total <= 0:
        raise ValueError(f"La somma dei pesi {weight_key!r} deve essere positiva.")
    return weights / total


def class_aware_weights(class_counts: np.ndarray) -> np.ndarray:
    """Calcola pesi cliente che danno massa uguale a ogni classe presente.

    Per la classe ``k``, i client ricevono insieme massa 1/K ripartita in
    proporzione ai loro esempi ``n_ik``. Le K classi sono quelle osservate nel
    round; l'output è normalizzato e può pesare direttamente i modelli locali.
    """
    if class_counts.ndim != 2 or class_counts.shape[0] == 0:
        raise ValueError("class_counts deve avere forma (numero_client, numero_classi).")
    if np.any(class_counts < 0):
        raise ValueError("I conteggi delle classi non possono essere negativi.")

    totals = class_counts.sum(axis=0)
    present = totals > 0
    if not np.any(present):
        raise ValueError("Nessun campione disponibile per l'aggregazione class-aware.")

    # La normalizzazione finale rende la formula valida anche quando una classe
    # non compare nei client selezionati di uno specifico round.
    raw = (class_counts[:, present] / totals[present]).sum(axis=1)
    return raw / raw.sum()


def fednova_coefficients(sample_weights: np.ndarray, local_steps: np.ndarray) -> tuple[np.ndarray, float]:
    """Restituisce coefficienti FedNova e numero effettivo di passi.

    Con SGD senza momentum (richiesto dalla variante FedNova di ``local_train``), il normalizzatore FedNova
    ``a_i`` coincide con il numero di aggiornamenti dell'ottimizzatore locale.
    """
    if sample_weights.shape != local_steps.shape:
        raise ValueError("sample_weights e local_steps devono avere la stessa forma.")
    if np.any(local_steps <= 0):
        raise ValueError("FedNova richiede almeno un batch locale per ogni client.")
    tau_eff = float(np.dot(sample_weights, local_steps))
    return tau_eff * sample_weights / local_steps, tau_eff


def _weighted_model_average(records: list[RecordDict], weights: np.ndarray) -> ArrayRecord:
    """Media pesata dei modelli restituiti dai client."""
    aggregated: dict[str, np.ndarray] = {}
    dtypes: dict[str, np.dtype] = {}
    for record, weight in zip(records, weights, strict=True):
        arrays = next(iter(record.array_records.values()))
        for key, value in arrays.items():
            array = value.numpy()
            if key not in aggregated:
                aggregated[key] = np.asarray(array, dtype=np.float64) * weight
                dtypes[key] = array.dtype
            else:
                aggregated[key] += np.asarray(array, dtype=np.float64) * weight
    return ArrayRecord(
        {key: Array(value.astype(dtypes[key], copy=False)) for key, value in aggregated.items()}
    )


def _fednova_model_update(
    records: list[RecordDict],
    coefficients: np.ndarray,
    reference: dict[str, np.ndarray],
) -> ArrayRecord:
    """Applica l'aggiornamento FedNova normalizzato al modello del server."""
    update: dict[str, np.ndarray] = {
        key: np.zeros_like(value, dtype=np.float64) for key, value in reference.items()
    }
    for record, coefficient in zip(records, coefficients, strict=True):
        arrays = next(iter(record.array_records.values()))
        if set(arrays.keys()) != set(reference):
            raise ValueError("I parametri dei client non coincidono con il modello server.")
        for key, value in arrays.items():
            update[key] += coefficient * (
                np.asarray(value.numpy(), dtype=np.float64) - reference[key]
            )
    return ArrayRecord(
        {
            key: Array((reference[key] + update[key]).astype(reference[key].dtype, copy=False))
            for key in reference
        }
    )


class FedNova(FedAvg):
    """FedNova per SGD senza momentum, compatibile con la pipeline corrente.

    La strategia normalizza i delta locali per ``num_batches`` e poi applica il
    numero effettivo di passi pesato per esempi. Questo rimuove il bias dovuto
    agli shard CESNET di dimensioni diverse, pur mantenendo gli stessi client,
    loader, criterio e salvataggi delle altre baseline.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._reference_arrays: dict[str, np.ndarray] | None = None

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        self._reference_arrays = {
            key: value.numpy().copy() for key, value in arrays.items()
        }
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid_replies:
            return None, None
        if self._reference_arrays is None:
            raise RuntimeError("FedNova non ha il modello server di riferimento.")

        records = [message.content for message in valid_replies]
        sample_weights = _sample_weights(records, self.weighted_by_key)
        local_steps = np.asarray(
            [float(_metric_record(record)[LOCAL_STEPS_KEY]) for record in records],
            dtype=np.float64,
        )
        coefficients, tau_eff = fednova_coefficients(sample_weights, local_steps)
        arrays = _fednova_model_update(records, coefficients, self._reference_arrays)
        metrics = self.train_metrics_aggr_fn(records, self.weighted_by_key)
        metrics["fednova-effective-steps"] = tau_eff
        log.info("FedNova round %d | passi locali effettivi %.2f", server_round, tau_eff)
        return arrays, metrics


class ClassAwareFedAvg(FedAvg):
    """FedAvg che attribuisce identica massa aggregata a ciascuna classe."""

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid_replies:
            return None, None

        records = [message.content for message in valid_replies]
        counts = np.asarray(
            [_metric_record(record)[TRAIN_CLASS_COUNTS_KEY] for record in records],
            dtype=np.float64,
        )
        weights = class_aware_weights(counts)
        arrays = _weighted_model_average(records, weights)
        metrics = self.train_metrics_aggr_fn(records, self.weighted_by_key)
        metrics["classaware-max-client-weight"] = float(weights.max())
        metrics["classaware-min-client-weight"] = float(weights.min())
        log.info(
            "ClassAwareFedAvg round %d | pesi client min %.4f max %.4f",
            server_round,
            weights.min(),
            weights.max(),
        )
        return arrays, metrics
