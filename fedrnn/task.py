# modello, addestramento locale, valutazione e metriche
# modificato a partire da task.py del template flwr

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fedrnn.config import CLASS_NAMES, NUM_CLASSES, NUM_FEATURES


# modello centralizzato DeviceRNN 
class DeviceRNN(nn.Module):
    def __init__(
        self,
        c_in: int = NUM_FEATURES,
        c_out: int = NUM_CLASSES,
        hidden_size: int = 128,
        n_layers: int = 2,
        bidirectional: bool = True,
        rnn_dropout: float = 0.2,
        fc_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=c_in,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=rnn_dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), c_out)
        self.apply(self._init_weights)

    # Inizializzazione
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        for name, param in module.named_parameters(recurse=False):
            if "weight_ih" in name:
                nn.init.xavier_normal_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(2, 1)
        output, _ = self.rnn(x)
        output = output[:, -1]
        return self.fc(self.dropout(output))


# su quale dispositivo si allena
def pick_device() -> torch.device:
    if os.getenv("FEDRNN_FORCE_CPU", "").strip().lower() in {"1", "true", "yes"}:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# per calcolare i pesi, uso i conteggi aggregati raccolti nel round di statistiche 
def balanced_class_weights(
    class_counts: np.ndarray,
    device: torch.device | None = None,
) -> torch.Tensor:
    # pesi inversamente proporzionali alla frequenza delle classi
    counts = np.asarray(class_counts, dtype=np.float64)
    n_classes = len(counts)
    total = counts.sum()
    # una classe assente prende peso 1 invece di infinito
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(counts > 0, total / (n_classes * np.maximum(counts, 1)), 1.0)
    tensor = torch.tensor(weights, dtype=torch.float32)
    return tensor.to(device) if device is not None else tensor


# round di addestramento locale dove il client riceve i pesi globali
# le scelte sono prese dal modello centralizzato di base a meno che non specificato
def local_train(
    model: nn.Module,
    loader: DataLoader,
    *,
    epochs: int,
    lr: float, # viene scelto dal server
    weight_decay: float,
    class_weights: torch.Tensor,
    device: torch.device,
    max_grad_norm: float = 1.0,
    proximal_mu: float = 0.0,
) -> dict[str, float]:
    # addestra sui dati locali e restituisce la loss media e i batch visti
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # proximal_mu di Fedprox ha bisogno dei pesi globali con cui il round è iniziato 
    # mentre per le altre strategie non lo richiedono
    global_params = (
        [p.detach().clone() for p in model.parameters()] if proximal_mu > 0 else None
    )

    running_loss = 0.0
    n_batches = 0
    for _ in range(epochs):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X), y)

            if global_params is not None:
                # lo scarto serve per penalizzare i client man mano che il modello si allontana, quindi le due epoche locali lo spostano meno
                scarto = sum(
                    ((locale - globale) ** 2).sum()
                    for locale, globale in zip(model.parameters(), global_params)
                )
                loss = loss + (proximal_mu / 2.0) * scarto

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            running_loss += float(loss.item())
            n_batches += 1

    return {
        "train_loss": running_loss / max(n_batches, 1),
        "num_batches": float(n_batches),
    }


# valutazione locale sul client e per il test sul server

# restituisce loss e matrice di confusione
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    class_weights: torch.Tensor,
    device: torch.device,
) -> tuple[float, np.ndarray, int]:
    # restituisce (loss media, matrice di confusione, numero di campioni)
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    total_loss = 0.0
    n_batches = 0
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        total_loss += float(criterion(logits, y).item())
        n_batches += 1
        y_true.append(y.cpu().numpy())
        y_pred.append(logits.argmax(dim=1).cpu().numpy())

    if n_batches == 0:
        return 0.0, np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64), 0

    true = np.concatenate(y_true)
    pred = np.concatenate(y_pred)
    return total_loss / n_batches, confusion_matrix(true, pred), len(true)


# invio la matrice di confusione anzichè la f1 score perchè la f1 score si ricava dalla matrice di confusione e per ottenere la f1 score totale
# basta effettuare il calcolo sul server una volta combinate tutte le matrici di confusione provenienti dai client
def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    flat = y_true.astype(np.int64) * num_classes + y_pred.astype(np.int64)
    counts = np.bincount(flat, minlength=num_classes * num_classes) # la preferisco a quella di scikit-learn perchè sono due righe
        # e non ripassa dai suoi controlli sugli array, che qui si pagherebbero a ogni round
    return counts.reshape(num_classes, num_classes)


# utilizzo le stesse metriche del progetto centralizzato in modo che il confronto si possa fare riga per riga: accuratezza, macro f1, f1 pesata,
# accuratezza bilanciata e f1 per ciascuna classe (il riferimento rimane comunque la macro-f1)
def metrics_from_confusion(cm: np.ndarray) -> dict[str, float]:
    """Metriche scalari, pronte per essere messe in un MetricRecord."""
    cm = np.asarray(cm, dtype=np.float64)

    support = cm.sum(axis=1)        # quanti esempi veri per classe
    predicted = cm.sum(axis=0)      # quante predizioni per classe
    tp = np.diag(cm)
    total = cm.sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)

    # l'accuratezza bilanciata è la media dei recall, e la calcolo solo sulle classi che compaiono davvero quindi
    # una classe che nella matrice non c'è avrebbe recall 0 e mi tirerebbe giù la media senza dire niente sul modello
    present = support > 0
    balanced_accuracy = float(recall[present].mean()) if present.any() else 0.0

    weights = support / total if total > 0 else np.zeros_like(support)

    out: dict[str, float] = {
        "accuracy": float(tp.sum() / total) if total > 0 else 0.0,
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * weights).sum()),
        "balanced_accuracy": balanced_accuracy,
    }
    for idx, name in enumerate(CLASS_NAMES):
        key = name.replace("-", "_")
        out[f"f1_{key}"] = float(f1[idx])
        out[f"recall_{key}"] = float(recall[idx])
        out[f"precision_{key}"] = float(precision[idx])
        out[f"support_{key}"] = float(support[idx])
    return out


# visto che il MetricRecord di flwr accetta scalari o liste di scalari, la matrice viene appiattita in 9 interi e ricostruita dal server

# appiattimento
def flatten_confusion(cm: np.ndarray) -> list[int]:
    return [int(v) for v in np.asarray(cm).reshape(-1)]

# ricostruzione
def unflatten_confusion(values, num_classes: int = NUM_CLASSES) -> np.ndarray:
    return np.asarray(values, dtype=np.int64).reshape(num_classes, num_classes)


def format_confusion(cm: np.ndarray) -> str:
    # matrice di confusione leggibile nel log della serverapp
    cm = np.asarray(cm, dtype=np.int64)
    width = max(len(name) for name in CLASS_NAMES)
    header = " " * (width + 2) + "".join(f"{name[:10]:>12}" for name in CLASS_NAMES)
    lines = [header, " " * (width + 2) + "-" * (12 * NUM_CLASSES)]
    for idx, name in enumerate(CLASS_NAMES):
        row = "".join(f"{int(v):>12d}" for v in cm[idx])
        lines.append(f"{name:>{width}} |{row}")
    return "\n".join(lines)
