# CLASSIFICAZIONE FEDERATA DEL TIPO DI DISPOSITIVO

Il progetto riguarda una rete neurale ricorrente addestrata con [Flower](https://flower.ai) sul
dataset [CESNET-TimeSeries24](https://www.nature.com/articles/s41597-025-04603-x), per riconoscere che tipo di dispositivo si nasconde dietro un indirizzo IP guardando solo il traffico che genera.

## Cosa fa
Ciascun indirizzo IP della rete produce una serie temporale.
Per 280 giorni si registrano 18 features del suo traffico: pacchetti, byte, destinazioni
contattate, rapporto TCP/UDP e durata media delle connessioni. Da questo comportamento,
il modello indovina se l'indirizzo appartiene a un **end-device** (computer, telefoni),
a un **server** o a un **net-device** (router, firewall).

Ogni subnet istituzionale è un client che si addestra sui propri dati e invia verso 
l'esterno solo i pesi della rete neurale locale.

La federazione è fatta di **69 client**, uno per subnet, con 82.527 indirizzi
distribuiti fra loro e 9.238 tenuti da parte dal server per la valutazione
finale.

| | macro-F1 |
|---|---|
| RNN centralizzata (progetto di partenza) | 0,7615 |
| RNN federata, migliore configurazione | 0,6286 |

## Come avviarlo

**Requisiti di Base**:
* Python compreso tra 3.10 e 3.13
* 11 GB liberi sul disco
* GPU NVIDIA (facoltativo, in alternativa usare il processore)


```bash
git clone https://github.com/idkleti/federated-rnn-cesnet-flwr.git
cd federated-rnn-cesnet-flwr

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python prepare_data.py     # scarica 3,2 GB e scrive i 69 shard (la prima volta)
flwr run . --stream
python tools/grafici.py
```

Gli iperparametri si cambiano in `pyproject.toml` oppure da linea di comando:

```bash
flwr run . --run-config "num-server-rounds=100 strategy=\"fedadam\""
```

**Se la simulazione viene interrotta per esaurimento di memoria**, alza
`num-cpus` in `pyproject.toml`: più core per client significano meno client
contemporaneamente. 
I valori attuali sono tarati su una macchina con 16 core e 7,4 GB di RAM.

## Struttura del progetto

```
README.md              <-- questo file
RISULTATI.md           risultati delle analisi dell'app

pyproject.toml         configurazione di flwr
requirements.txt       requisiti delle librerie

prepare_data.py        scarica il dataset, divide gli indirizzi per subnet
                       e scrive uno shard per client
fedrnn/
  config.py            costanti condivise: classi, feature, seed, percorsi
  task.py              il modello, l'addestramento locale, le metriche
  data.py              lettura degli shard e costruzione dei DataLoader
  client_app.py        cosa gira sul nodo che possiede i dati
  server_app.py        cosa gira sul nodo che coordina la federazione
tools/
  esperimenti.sh       i comandi che hanno prodotto i risultati
  grafici.py           le figure, rigenerate dagli storici delle run
outputs/               storici delle run e figure
```

## Risultati
All'interno del file **[RISULTATI.md](RISULTATI.md)** si trovano 15 esecuzioni complete (due per configurazione), con il confronto tra strategie di aggregazione e l'analisi di dove il modello federato peggiora rispetto a quello centralizzato.

**In breve, la configurazione migliore è FedAvg con fraction-train = 0.5**.
Il divario di tredici punti della macro f1-score rispetto al centralizzato è dovuto alle classi **server** e **net-device**.

Per i **net-device**: la classe è concentrata per il 55% dentro un solo client, piccolo, e la media pesata di FedAvg la cancella nei round in cui quel client non partecipa.

Per i **server**: è la classe meglio distribuita di tutte (61 client su 69), quindi il divario è probabilmente causato dalla somiglianza con gli end-device (esperimento indicato in **[RISULTATI.md](RISULTATI.md)**) visto che il client più grande pesa il 21% della federazione ed è puro end-device al 99,8%.
