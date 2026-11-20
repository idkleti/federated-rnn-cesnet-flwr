# CESNET-TimeSeries24 federated device classification — Flower app

This is a **standalone code copy** for reproducing the experiments in the paper. It contains the Flower `ServerApp` and `ClientApp`, the RNN and aggregation strategies, and the dataset-preparation script. It intentionally contains **no SLURM scripts, plotting tools, dataset, shards, models, or results**. The dataset is obtained separately through `cesnet-tszoo` and is not redistributed here.

## Contents

- `pyproject.toml`: dependencies, Flower entry points, and default run settings.
- `fedrnn/server_app.py`: strategy selection, federated validation, checkpoint selection, and test evaluation.
- `fedrnn/client_app.py`: local statistics, training, and validation.
- `fedrnn/strategies.py`: FedNova and class-aware aggregation.
- `fedrnn/task.py`, `fedrnn/data.py`, `fedrnn/config.py`: model, data loading, and configuration.
- `prepare_data.py`: creates the fixed server test split and the 69 client shards.

## Requirements and data

Use Python 3.12 and install the project in an environment with access to PyPI:

```bash
python -m pip install -e .
```

The dependency versions are specified in `pyproject.toml` (notably Flower 1.32.1). On an HPC cluster, prepare the environment and download the dataset only where permitted. **Never run Flower simulations on a login node**; use an allocated compute node. The commands below are intended to be typed manually from this directory on a suitable machine or compute allocation.

Download and validate the CESNET data once, then build the natural subnet partition:

```bash
python prepare_data.py --download-only
python prepare_data.py --offline --partition subnet
```

`--download-only` needs network access. If data were downloaded elsewhere, copy the complete `time_dataset/` directory, including its `.fedrnn-cesnet-ready.json` manifest, into this directory before using `--offline`. By default, `prepare_data.py` writes 69 client files and `meta.json` in `shards/`, plus a fixed server test set. The script reports the actual number of clients; the examples below assume the default 69. Data and output paths may instead be set with `CESNET_DATA_ROOT`, `FEDRNN_SHARD_ROOT`, and `FEDRNN_OUTPUT_ROOT` (absolute paths recommended).

**Important:** every new invocation of `prepare_data.py` rebuilds the client shards. Finish all runs for one partition before switching to another. Do not run experiments concurrently while rebuilding shards. Results are saved in `outputs/` as `history_*.json` and model `.pt` files; each history records the partition and run configuration.

## Run manually

Flower 1.32.1 requires the number of simulated SuperNodes in `--federation-config`, not in `pyproject.toml`. On the H100 setup used for the paper, the federation configuration was:

```bash
FEDERATION_CONFIG='num-supernodes=69 client-resources-num-cpus=11 client-resources-num-gpus=0.5'
COMMON='num-server-rounds=25 fraction-train=0.5 fraction-evaluate=0.5 lr-decay=0.97 evaluate-central-test=true'
DEFAULT="$COMMON local-epochs=2"
```

Use resource values appropriate to your machine. For CPU-only execution, use `client-resources-num-gpus=0` and adjust the CPU allocation; keep `num-supernodes` equal to the count printed by `prepare_data.py`.

For the **natural non-IID subnet partition**, first run `python prepare_data.py --offline --partition subnet`, then execute any of these configurations:

```bash
flwr run . --federation-config "$FEDERATION_CONFIG" --run-config "$DEFAULT strategy=\"fedavg\""
flwr run . --federation-config "$FEDERATION_CONFIG" --run-config "$DEFAULT strategy=\"fedprox\" proximal-mu=1.0"
flwr run . --federation-config "$FEDERATION_CONFIG" --run-config "$COMMON strategy=\"fedyogi\" local-epochs=1 server-learning-rate=0.01"
flwr run . --federation-config "$FEDERATION_CONFIG" --run-config "$DEFAULT strategy=\"classaware\""
flwr run . --federation-config "$FEDERATION_CONFIG" --run-config "$DEFAULT strategy=\"fednova\" local-optimizer=\"sgd\""
flwr run . --federation-config "$FEDERATION_CONFIG" --run-config "$DEFAULT strategy=\"fedavgm\" server-learning-rate=1.0 server-momentum=0.9"
flwr run . --federation-config "$FEDERATION_CONFIG" --run-config "$DEFAULT strategy=\"fedadagrad\" server-learning-rate=0.1 server-tau=0.001"
```

Run each command **five times** for the five-replicate comparison. The paper used local Adam for all strategies except FedNova, which uses plain local SGD. To reproduce the local-epoch ablation, use the same non-IID shards and FedAvg/FedProx commands, setting `local-epochs` to `1`, `2`, and `4`; repeat each combination five times. An example is:

```bash
flwr run . --federation-config "$FEDERATION_CONFIG" --run-config "$COMMON strategy=\"fedprox\" proximal-mu=1.0 local-epochs=4"
```

The IID control preserves the real client sizes but shuffles addresses. Rebuild the shards and run FedAvg:

```bash
python prepare_data.py --offline --partition random-sizes
flwr run . --federation-config "$FEDERATION_CONFIG" --run-config "$DEFAULT strategy=\"fedavg\""
```

The shared-set control adds 500 examples **per class** to each subnet client:

```bash
python prepare_data.py --offline --partition subnet --mini-dataset 500
flwr run . --federation-config "$FEDERATION_CONFIG" --run-config "$DEFAULT strategy=\"fedavg\""
```

Restore the natural partition with `python prepare_data.py --offline --partition subnet`. Other supported variants can be run manually by changing `--run-config` or `prepare_data.py` arguments: `strategy=\"fedadam\"`, `--partition random` (equal client sizes), `--rebalance-net-device N`, different participation fractions, rounds, decay, and FedProx `proximal-mu`. For a validation-only hyperparameter sweep, set `evaluate-central-test=false`; use the test set only for the final, preselected configurations.

The checkpoint is selected by federated validation macro-F1. With `evaluate-central-test=true`, the central test set is evaluated once after checkpoint selection, not used to choose the checkpoint or hyperparameters. The experiment is a local Flower simulation of 69 clients, **not** a deployment on 69 physical machines.

## Sharing this folder

From the parent directory, create an archive containing only the reviewed code files:

```bash
zip -r conference_flower.zip conference_flower/pyproject.toml conference_flower/prepare_data.py conference_flower/README.md conference_flower/LICENSE conference_flower/.gitignore conference_flower/fedrnn -x '*/__pycache__/*' '*.pyc'
```

This excludes `time_dataset/`, `shards/`, and `outputs/` even if you have already downloaded data or run experiments. The `.gitignore` excludes those directories from Git and the Flower app bundle, but a generic ZIP command does not automatically honor `.gitignore`.
