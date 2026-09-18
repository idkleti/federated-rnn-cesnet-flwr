#!/usr/bin/env bash

# Prepara CESNET-TimeSeries24 su una macchina con Internet e lo usa su un
# cluster HPC isolato. Non sostituisce prepare_data.py né esperimenti.sh.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/.." && pwd -P)"
cd -- "$project_root"

usage() {
    cat <<'EOF'
Uso:
  bash tools/hpc_data.sh bundle [archivio.tar.gz]
  bash tools/hpc_data.sh install <archivio.tar.gz> <directory-dati>
  bash tools/hpc_data.sh shards <directory-dati> [opzioni di prepare_data.py]

Comandi:
  bundle   (macchina con Internet) scarica e crea un archivio trasferibile.
  install  (HPC) estrae l'archivio in una directory vuota.
  shards   (HPC) genera gli shard senza rete; le opzioni successive vengono
           passate a prepare_data.py, ad esempio --partition random-sizes.

Variabili opzionali:
  PYTHON_BIN          Interprete da usare (default: venv/bin/python, poi python3)
  CESNET_DATA_ROOT    Directory dati usata da bundle se diversa da time_dataset/

Esempio:
  # macchina con Internet
  bash tools/hpc_data.sh bundle /tmp/cesnet-time-dataset.tar.gz
  # trasferire l'archivio con il metodo consentito dal cluster
  # HPC
  bash tools/hpc_data.sh install /hpc/home/clmlnz/cesnet-time-dataset.tar.gz /hpc/home/clmlnz/federated-rnn-cesnet-flwr/time_dataset
  bash tools/hpc_data.sh shards /hpc/home/clmlnz/federated-rnn-cesnet-flwr/time_dataset --partition subnet
EOF
}

python_bin="${PYTHON_BIN:-}"
if [[ -z "$python_bin" ]]; then
    if [[ -x "$project_root/venv/bin/python" ]]; then
        python_bin="$project_root/venv/bin/python"
    else
        python_bin="python3"
    fi
fi

run_bundle() {
    local archive_input="${1:-$project_root/cesnet-time-dataset.tar.gz}"
    local data_root="${CESNET_DATA_ROOT:-$project_root/time_dataset}"
    local archive_dir archive

    mkdir -p -- "$(dirname -- "$data_root")"
    data_root="$(cd -- "$(dirname -- "$data_root")" && pwd -P)/$(basename -- "$data_root")"
    mkdir -p -- "$(dirname -- "$archive_input")"
    archive_dir="$(cd -- "$(dirname -- "$archive_input")" && pwd -P)"
    archive="$archive_dir/$(basename -- "$archive_input")"

    if [[ -e "$archive" ]]; then
        echo "Archivio già esistente: $archive" >&2
        echo "Scegli un nome diverso per evitare di sovrascriverlo." >&2
        return 1
    fi

    echo "==> Materializzo il dataset in $data_root"
    CESNET_DATA_ROOT="$data_root" "$python_bin" prepare_data.py --download-only

    echo "==> Creo $archive"
    tar -C "$(dirname -- "$data_root")" -czf "$archive" "$(basename -- "$data_root")"
    echo "Bundle pronto: $archive"
}

validate_archive() {
    local archive="$1"
    local entry top=""

    while IFS= read -r entry; do
        [[ -n "$entry" ]] || continue
        if [[ "$entry" == /* || "$entry" == *"../"* || "$entry" == ".." ]]; then
            echo "Archivio non sicuro: contiene il percorso $entry" >&2
            return 1
        fi
        local entry_top="${entry%%/*}"
        if [[ -z "$top" ]]; then
            top="$entry_top"
        elif [[ "$entry_top" != "$top" ]]; then
            echo "Archivio non valido: contiene più directory radice." >&2
            return 1
        fi
    done < <(tar -tzf "$archive")

    [[ -n "$top" ]] || { echo "Archivio vuoto: $archive" >&2; return 1; }
}

run_install() {
    local archive="$1"
    local data_root="$2"

    [[ -f "$archive" ]] || { echo "Archivio non trovato: $archive" >&2; return 1; }
    validate_archive "$archive"
    if [[ -e "$data_root" ]] && [[ -n "$(find "$data_root" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "La directory di destinazione non è vuota: $data_root" >&2
        echo "Per sicurezza non estraggo sopra file esistenti." >&2
        return 1
    fi
    mkdir -p -- "$data_root"

    echo "==> Estraggo il dataset in $data_root"
    tar -xzf "$archive" -C "$data_root" --strip-components=1 --no-same-owner
    [[ -f "$data_root/.fedrnn-cesnet-ready.json" ]] || {
        echo "Archivio non valido: manca il manifest offline." >&2
        return 1
    }
    echo "Dataset installato: $data_root"
}

run_shards() {
    local data_root="$1"
    shift
    [[ -d "$data_root" ]] || { echo "Directory dati non trovata: $data_root" >&2; return 1; }
    echo "==> Genero shard offline da $data_root"
    CESNET_DATA_ROOT="$data_root" "$python_bin" prepare_data.py --offline "$@"
}

case "${1:-}" in
    bundle) shift; [[ $# -le 1 ]] || { usage >&2; exit 2; }; run_bundle "${1:-}" ;;
    install) shift; [[ $# -eq 2 ]] || { usage >&2; exit 2; }; run_install "$1" "$2" ;;
    shards) shift; [[ $# -ge 1 ]] || { usage >&2; exit 2; }; run_shards "$@" ;;
    -h|--help|help) usage ;;
    *) usage >&2; exit 2 ;;
esac
