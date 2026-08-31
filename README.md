# CLASSIFICAZIONE FEDERATA DEL TIPO DI DISPOSITIVO
Il file usa la convenzione numerica americana (1,234.56).

Il progetto riguarda una rete neurale ricorrente addestrata con [Flower](https://flower.ai) sul
dataset [CESNET-TimeSeries24](https://www.nature.com/articles/s41597-025-04603-x), per riconoscere che tipo di dispositivo si nasconde dietro un indirizzo IP guardando solo il traffico che genera.

## Cosa fa
Ciascun indirizzo IP della rete produce una serie temporale.
Per 280 giorni si registrano 18 features del suo traffico: pacchetti, byte, destinazioni contattate, rapporto TCP/UDP e durata media delle connessioni. 
Da questo comportamento, il modello indovina se l'indirizzo appartiene a un **end-device** (computer, telefoni), a un **server** o a un **net-device** (router, firewall).

Ogni subnet istituzionale è un client che si addestra sui propri dati e invia verso l'esterno solo i pesi della rete neurale locale.

La federazione è fatta di **69 client**, uno per subnet, con 82,527 indirizzi distribuiti fra loro e 9,238 tenuti da parte dal server per la valutazione finale.

La presenza dei dati di test sul server è solo per scopo "educativo" per capire quanto si perde a federare. In una situazione reale, il server non contiene dati di test.

| | macro-F1 |
|---|---|
| RNN centralizzata (progetto di partenza) | 0.7615 |
| RNN federata, migliore configurazione | 0.6310 |

## Come avviarlo

**Requisiti minimi**:
* Python compreso tra 3.10 e 3.13
* 11 GB liberi sul disco
* GPU NVIDIA (facoltativo, in alternativa usare il processore)

```bash
git clone https://github.com/idkleti/federated-rnn-cesnet-flwr.git
cd federated-rnn-cesnet-flwr

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python prepare_data.py     # scarica 3.2 GB e scrive i 69 shard (la prima volta)
flwr run . --stream        # aggiungere PYTHONUNBUFFERED=1 qualora il log non venisse mostrato correttamente ad ogni run
python tools/grafici.py
```

Gli iperparametri si cambiano in `pyproject.toml` oppure da linea di comando:

```bash
flwr run . --run-config "num-server-rounds=100 strategy=\"fedadam\""
flwr run . --run-config "strategy=\"fedprox\" proximal-mu=0.8"    # la configurazione migliore
```

Il modo in cui gli indirizzi vengono divisi fra i client si sceglie invece in `prepare_data.py`:

```bash
python prepare_data.py --partition random          # indirizzi mescolati, client uguali
python prepare_data.py --partition random-sizes    # mescolati, dimensioni delle subnet vere
python prepare_data.py --rebalance-net-device 10   # copie della classe rara a chi non ne ha
```

Servono a misurare quanto del divario dipende dalla forma della federazione (vedi **[RISULTATI.md](RISULTATI.md)**).

**Se la simulazione viene interrotta per esaurimento di memoria**, alza `num-cpus` in `pyproject.toml`.
I valori attuali sono tarati su una macchina con 16 core e 7.4 GB di RAM.

**Se la simulazione viene interrotta e non viene mostrato alcun errore in console**, killare tutti i processi di flower e riavviare la simulazione.

## Struttura del progetto

```
README.md              <-- questo file
RISULTATI.md           risultati delle analisi dell'app

pyproject.toml         configurazione di flwr
requirements.txt       requisiti delle librerie

prepare_data.py        scarica il dataset, divide gli indirizzi fra i client (per subnet o a caso) e scrive uno shard per ciascuno
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

All'interno del file **[RISULTATI.md](RISULTATI.md)** si trovano **350 esecuzioni complete, 25 per ogni configurazione**, con il confronto tra strategie di aggregazione, quattro modi diversi di dividere gli indirizzi fra i client, e l'analisi di dove il modello federato peggiora rispetto a quello centralizzato.

**In breve, la configurazione migliore è FedProx con proximal-mu = 0.8, fraction-train = 0.5 e lr-decay = 0.97**, con f1-score 0.6310 contro lo 0.7615 del centralizzato. Il divario è dovuto per quasi tre quinti alla classe **server** e per poco più di un terzo ai **net-device**. La differenza fra `fraction-train` 0.3 e 0.5 non esiste e il decadimento del learning rate aiuta invece di peggiorare.

La prova su `proximal-mu` è stata effettuata per ultima, dopo tutte le prove precedenti, per utilizzare una strategia dedicata ai dati non-IID, cambiando il valore di mu. Portarlo a 0.8 sposta di sette centesimi la F1 dei `server`.

**Il risultato principale** è che il divario dipende da **come le classi sono distribuite fra i client**, e non da quanto i client sono grandi. Mescolando gli indirizzi fra i 69 client ma tenendo le dimensioni vere delle subnet si recupera il 44% del divario; se invece si rendono i client tutti della stessa dimensione il risultato peggiora.

Da qua si deriva il perchè i `server` sono la classe più penalizzata pur essendo presenti in 61 client su 69, ovvero che non conta in quanti client una classe compare ma in che proporzione compare dentro ciascuno.
Aggiungere qualche net-device ai client che ne hanno pochi non basta (provato con 425 serie), perchè lascia intatte le proporzioni fra le classi dentro ogni client. L'unico intervento che ha spostato davvero i `server` è stato trattenere i client vicino al modello globale durante le epoche locali, cioè alzare `proximal-mu`.