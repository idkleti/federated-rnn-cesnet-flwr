# RISULTATI DEGLI ESPERIMENTI
Il file usa la convenzione numerica americana (1,234.56).

Sono state effettuate **400 esecuzioni complete** su WSL2 (16 core, 7.4 GB di RAM, RTX 5060 con 8 GB), per un totale di circa 47 ore di solo addestramento.
I comandi che le hanno prodotte si trovano in [`tools/esperimenti.sh`](tools/esperimenti.sh), gli storici round per round in `outputs/history_*.json`, e le figure si rigenerano con `python tools/grafici.py`.

**Ogni configurazione è stata eseguita 25 volte.**

## Confronti

Il riferimento è la RNN centralizzata, addestrata 25 volte sulle stesse serie di addestramento e validazione della federazione, che ottiene in media **macro-F1 0.7355** sul test set.

Il confronto viene effettuato sullo stesso test set poichè lo split avviene con lo stesso seed (111) e le stesse proporzioni, ottenendo gli stessi conteggi per classe (8,136 end-device, 218 net-device e 884 server).

Il modello inviato al server viene selezionato in base alla macro-f1 score migliore della run sui dati di validazione dei client, senza mai guardare il test. Si utilizza come metrica la macro-f1 score rispetto all'accuratezza poichè bisogna tenere conto dello sbilanciamento delle classi nel dataset.

Il test set del server è lo stesso identico in tutte e cinque le partizioni confrontate più sotto. La divisione di holdout avviene prima di distribuire gli indirizzi ai client, quindi non dipende da come poi vengono divisi, e `prepare_data.py` se lo ritrova già scritto per poi riconoscerlo dagli identificatori e tenerselo.

## La federazione

Un client corrisponde ad una subnet istituzionale con almeno 10 indirizzi etichettati.

| | |
|---|---|
| client | 69 |
| indirizzi distribuiti | 82,527 |
| indirizzi al server (test) | 9,238 |
| per client: min / mediana / max | 10 / 34 / 17,398 |
| pesi di classe aggregati | 0.3761 / 15.0014 / 3.6447 |

![struttura della federazione](outputs/figure/05_partizione.png)

## Tutti gli esperimenti sulla federazione vera

I risultati sono raggruppati in una tabella che indica (in ordine):

* **strategia**: strategia di aggregazione utilizzata
* **ft**: frazione di client che partecipa a ogni round
* **lrd**: fattore di decadimento del learning rate
* **mu**: forza del richiamo verso il modello globale, solo per FedProx
* **round**: numero di round effettuati nell'addestramento
* **consegnato**: macro-f1 sul test del modello selezionato
* **stab.**: quanto oscilla la macro-f1 negli ultimi 31 round
* **net-dev sotto 0.01**: in quanti degli ultimi 31 round la F1 dei net-device era praticamente azzerata

In tutti gli esperimenti si hanno 2 epoche locali, Adam a 6.92e-4, weight decay 0.01, batch 64.

Nel resto del documento le differenze fra configurazioni sono espresse in **x volte l'errore**, che si calcola dividendo la differenza fra le due medie per l'errore sulla differenza `√(e₁² + e₂²)`, dove `e = s / √25` per ciascuna configurazione e `s` è la deviazione standard delle sue 25 esecuzioni. Sotto 2 la differenza non si distingue dal caso.

| strategia | ft | lrd | mu | round | consegnato | stab. | net-dev sotto 0.01 |
|---|---|---|---|---|---|---|---|
| **fedprox** | 0.5 | 0.97 | **0.8** | 50 | **0.6310** | 0.0566 | 5/31 |
| fedprox | 0.5 | 0.97 | 1.0 | 50 | 0.6263 | 0.0504 | 2/31 |
| fedprox | 0.5 | 0.97 | 0.5 | 50 | 0.6214 | 0.0549 | 5/31 |
| fedprox | 0.5 | 0.97 | 0.2 | 50 | 0.6212 | 0.0538 | 9/31 |
| fedavg | 0.5 | 0.97 | — | 50 | 0.6101 | 0.0575 | 13/31 |
| fedprox | 0.5 | 0.97 | 0.1 | 50 | 0.6035 | 0.0501 | 11/31 |
| fedavg | 0.5 | — | — | 100 | 0.5983 | 0.0570 | 14/31 |
| fedavg | 0.3 | — | — | 50 | 0.5911 | 0.0789 | 14/31 |
| fedavg | 0.5 | — | — | 50 | 0.5910 | 0.0577 | 12/31 |
| fedadam | 0.5 | 0.97 | — | 50 | 0.5856 | 0.0587 | 9/31 |
| fedyogi | 0.5 | 0.97 | — | 50 | 0.5782 | 0.0545 | 14/31 |
| fedavg | 1.0 | — | — | 50 | 0.5249 | **0.0175** | **2/31** |
| *centralizzata* | | | | | *0.7355* | | |

![confronto fra le configurazioni](outputs/figure/02_confronto_strategie.png)

![curve di apprendimento](outputs/figure/01_curve_macro_f1.png)

## Le strategie adattive lato server non aiutano

A parità di partecipazione e decadimento:

| | consegnato | dev std |
|---|---|---|
| fedavg | 0.6101 | ± 0.0299 |
| fedprox (mu=0.1) | 0.6035 | ± 0.0421 |
| fedadam | 0.5856 | ± 0.0417 |
| fedyogi | 0.5782 | ± 0.0454 |

FedAvg e FedProx sono indistinguibili, e fra i due FedAvg ha la deviazione standard minore quindi è meno imprevedibile.

FedAdam e FedYogi, che modificano i pesi globali con un passo adattivo invece di mediarli e basta, stanno **sotto FedAvg di 2.94 volte l'errore** (il confronto è con FedYogi, la più bassa). Fra loro due invece la differenza è 0.60 volte, quindi sono la stessa cosa.

Alzando `proximal-mu`, FedProx si stacca da FedAvg (vedi sezione *Il richiamo di FedProx verso il modello globale*), e dimostra che quello che aiuta è trattenere i client e non adattare il server.


## Il decadimento del learning rate aiuta

| | consegnato |
|---|---|
| `lr-decay` 1.00 | 0.5910 ± 0.0362 |
| `lr-decay` 0.97 | **0.6101 ± 0.0299** |

Il guadagno è di **0.0192**, cioè 2.04 volte l'errore sulla media.
Con il decadimento l'ultimo passo vale il 22% del primo, quindi i round finali spostano meno i pesi e il modello smette di saltare da una parte all'altra proprio dove la selezione va a pescare.

## La partecipazione

| `fraction-train` | client per round | consegnato |
|---|---|---|
| 0.3 | 20 | 0.5911 ± 0.0258 |
| 0.5 | 34 | 0.5910 ± 0.0362 |
| 1.0 | 69 | 0.5249 ± 0.0313 |

Fra 0.3 e 0.5 la differenza è **−0.0001**, praticamente uguali.

Con la partecipazione totale invece il calo vale 0.066 nonostante sia la configurazione più stabile di tutte, infatti, la sua `stab.` vale 0.0175 contro 0.05-0.08 delle altre, e ha solo 2 round su 31 con i net-device azzerati. Da qui ricaviamo che un addestramento "instabile" produce dei round con macro-f1 score elevata fra cui scegliere, mentre uno stabile resta piatto e non offre niente di meglio.

## Quanto conta la forma della federazione

In questa sezione, la configurazione utilizzata è **FedAvg, fraction-train = 0.5, lr-decay 0.97 e 50 round** e l'unica cosa che cambia è la divisione degli indirizzi fra i client poichè era la migliore fra quelle provate fino a quel punto.
La prova su `proximal-mu` è stata fatta dopo questi confronti, tranne quello sull'insieme condiviso (eseguito più tardi, ma ancora con FedAvg apposta per restare confrontabile con le altre righe).
Gli indirizzi sono sempre 82,527 e i client sempre 69.

| | consegnato | min-max | divario |
|---|---|---|---|
| subnet, la federazione vera | 0.6101 ± 0.0299 | 0.5174-0.6515 | 0.1253 |
| subnet + 425 net-device condivisi | 0.6141 ± 0.0227 | 0.5366-0.6499 | 0.1214 |
| subnet + insieme condiviso 500 per classe | 0.5904 ± 0.0190 | 0.5600-0.6220 | 0.1451 |
| **casuale a dimensioni reali** | **0.6767 ± 0.0069** | 0.6609-0.6928 | **0.0588** |
| casuale a blocchi uguali | 0.6440 ± 0.0058 | 0.6325-0.6526 | 0.0915 |

Un paio di note sulla suddivisione casuale (usano gli stessi indirizzi e numero di client):
* **casuale a blocchi uguali**: gli indirizzi vengono mescolati e divisi in 69 pezzi identici
* **casuale a dimensioni reali**: mescolati allo stesso modo, ma le dimensioni restano quelle delle subnet vere, quindi resta il client da 17,398 indirizzi e restano quelli da 10

| | subnet | casuale | casuale a dimensioni reali |
|---|---|---|---|
| indirizzi per client | 10 / 34 / 17,398 | tutti ~1,196 | 10 / 34 / 17,398 |
| client con almeno un net-device | 42 su 69 | 69 su 69 | 43 su 69 |
| net-device di chi ne ha di più | 55.7% | 2.1% | 21.6% |

Guardando la seconda riga si può notare come, mantenendo le dimensioni vere ma mescolando i dati, **il fatto che quattro client su dieci non abbiano nemmeno un net-device dipende da quanto sono piccoli quei client e non da come sono fatte le subnet**.

La concentrazione del 55.7% invece crolla al 21.6%.

La federazione reale produce un risultato più basso e imprevedibile a seconda di come gira.

## Le classi contano, le dimensioni no

| passaggio | cosa cambia | effetto |
|---|---|---|
| subnet → casuale a dimensioni reali | si sparpagliano le **classi**, le dimensioni restano quelle vere | **+0.0666** (10.8 volte l'errore) |
| casuale a dimensioni reali → casuale a blocchi uguali | si pareggiano anche le **dimensioni** | **−0.0327** (18.1 volte l'errore) |

La prima divisione dei dati migliora il risultato mentre la seconda divisione peggiora il risultato totale, quindi una federazione di 69 client tutti uguali va peggio di una in cui pochi client sono enormi e molti minuscoli.

La spiegazione plausibile è che con pochi client grossi la media pesata di FedAvg sia dominata da aggiornamenti calcolati su molti dati, mentre 34 client da 1,200 indirizzi producono 34 aggiornamenti rumorosi da mediare fra loro.

Quindi dei **0.1253** di divario rispetto al modello centralizzato, che è quello di FedAvg fissato in questa sezione:
* **circa il 53% (0.0666)** dipende da come le classi sono distribuite fra le subnet
* il restante **47% (0.0588)** è il costo di federare in sé, che resta anche con la partizione migliore

## Prestare dati fra i client non aiuta

Le prove effettuate riguardavano il prestare poche serie della sola classe rara e il prestare a tutti un blocco condiviso bilanciato.

### Primo tentativo: solo net-device, a chi ne ha pochi

L'esperimento consisteva nel distribuire copie di net-device ai client che ne avevano pochi, fino a portarli tutti ad almeno dieci e senza duplicarle. Servono 425 serie, cioè lo 0.5% del pool, e i client senza nemmeno un net-device passano da 27 a zero. Con questa prova andiamo a **rilassare l'ipotesi del federated learning**, perchè delle serie passano da un client a un altro.

I donatori restano sempre sopra la soglia, quindi nessuno viene impoverito per arricchire un altro.

Il guadagno è **+0.0039** e la F1 dei net-device passa da 0.4969 a **0.4976**, che vuol dire poco.

### Secondo tentativo: un insieme bilanciato uguale per tutti

Sono state estratte **500 serie per classe** una volta sola e date a ciascun client. Sono 1,500 serie, ma contate una volta per client diventano 103,500 righe, di conseguenza la federazione si addestra su 184,527 righe invece di 82,527 e il condiviso corrisponde al **56%** di tutto.

Il consegnato scende da 0.6101 a **0.5904**, cioè **2.80 volte l'errore**, un peggioramento vero.

| classe | subnet | insieme condiviso |
|---|---|---|
| end-device | 0.9423 | **0.8924** |
| net-device | 0.4969 | 0.5104 |
| server | 0.3911 | 0.3683 |

Il motivo del peggioramento si trova nei **pesi di classe**, che il server calcola dai conteggi aggregati dai client. L'insieme condiviso, bilanciato e replicato, con le proporzioni aggregate 92,621 / 35,566 / 40,133, fa crollare i pesi (da 0.3761 / 15.0014 / 3.6447 a 0.6058 / 1.5775 / 1.3980).

| | end-device | net-device | server |
|---|---|---|---|
| subnet | 0.3761 | **15.0014** | 3.6447 |
| insieme condiviso | 0.6058 | **1.5775** | 1.3980 |

La classe rara perde il fattore 15 su cui si reggeva, e il modello impara un confine bilanciato che poi applica a un test set fatto di 8,136 end-device, 218 net-device e 884 server. Su quel test, sbilanciarsi verso le classi rare costa più di quanto renda.

L'addestramento invece diventa stabilissimo, infatti i round con i net-device azzerati passano da 12.8 su 31 a **zero**.

La conclusione di questa prova è che quello che serve non è che le classi siano bilanciate dentro ogni client, è che le proporzioni locali somiglino a quelle **globali**. Le partizioni casuali, che quelle proporzioni le rispettano, arrivano a 0.6767 (partizione casuale con dimensione reali).

### Perchè si è provato

L'idea viene da Zhao et al. 2018, *Federated Learning with Non-IID Data*, dove un piccolo insieme condiviso viene distribuito a tutti i partecipanti e si misura quanto si guadagna al crescere della frazione condivisa.

C'è da notare però come da loro l'insieme condiviso sia un dataset a parte che gli autori dichiarano non sensibile proprio perché non appartiene a nessun client (sta sul server e viene distribuito a tutti in fase di inizializzazione), mentre nel progetto le serie sono traffico reale di alcune istituzioni che finisce dentro altre istituzioni, quindi il prezzo in riservatezza è di natura diversa e va dichiarato.


### Note valide per entrambe le prove

* le 425 serie del primo tentativo sono **tutte diverse** senza duplicazione. Ogni serie finisce in due client soltanto, il suo proprietario e un ricevente, quindi il modello non può impararne a memoria poche ripetute ovunque
* le righe prestate stanno in coda allo shard e **non entrano mai nella validazione locale**. Se ci entrassero, il modello verrebbe valutato su una serie già vista in addestramento a casa di un altro client, e il punteggio salirebbe proprio sulle classi che l'esperimento vuole misurare
* nel secondo tentativo le serie condivise **escono anche dalle righe proprie di chi le possedeva**, per lo stesso motivo

## Il richiamo di FedProx verso il modello globale

Questa prova è stata effettuata dopo le prime 250 esecuzioni, ma prima di FedYogi e dell'insieme condiviso, per cercare di migliorare ulteriormente la F1 del modello, cambiando il valore di `proximal-mu`.

Il parametro pesa quanto un client viene penalizzato man mano che i suoi pesi si allontanano da quelli che il server gli ha spedito a inizio round. A 0 FedProx si comporta come FedAvg, e più sale meno i client possono allontanarsi dal modello globale durante le due epoche locali.

| mu | consegnato | stab. | net-dev sotto 0.01 |
|---|---|---|---|
| 0.1 | 0.6035 ± 0.0421 | 0.0501 | 11/31 |
| 0.2 | 0.6212 ± 0.0428 | 0.0538 | 9/31 |
| 0.5 | 0.6214 ± 0.0300 | 0.0549 | 5/31 |
| **0.8** | **0.6310 ± 0.0215** | 0.0566 | 5/31 |
| 1.0 | 0.6263 ± 0.0254 | 0.0504 | 2/31 |

A mu 0.8 il modello consegnato fa **0.6310** e porta il divario col centralizzato da 0.1253 a **0.1045**, con 2.83 volte l'errore di vantaggio su FedAvg. Fra 0.8 e 1.0 si equivalgono. Contro il default 0.1 si distinguono solo 0.8 e 1.0 (2.90 e 2.32 volte l'errore), mentre 0.2 e 0.5 si fermano a 1.47 e 1.73, sotto la soglia di 2.

Il guadagno cade quasi tutto su una classe sola:

| classe | fedavg | fedprox mu 0.8 | differenza |
|---|---|---|---|
| end-device | 0.9423 | 0.9446 | +0.0023 |
| net-device | 0.4969 | 0.4881 | −0.0088 |
| **server** | **0.3911** | **0.4601** | **+0.0689** |

Sono i `server`, cioè la classe che porta da sola la fetta più grossa del divario e che il prestito di net-device della sezione precedente non era riuscito a smuovere. I round in cui la F1 dei net-device è praticamente zero passano da 13 su 31 a 5.

## Dove si perde il divario, classe per classe

F1 per classe:

| classe | centralizzata | subnet (fedavg) | subnet (fedprox mu 0.8) | + net-device | + condiviso | dimensioni reali |
|---|---|---|---|---|---|---|
| end-device | 0.9602 | 0.9423 | 0.9446 | 0.9430 | 0.8924 | 0.9449 |
| net-device | 0.5885 | 0.4969 | 0.4881 | 0.4976 | 0.5104 | **0.5851** |
| server | 0.6577 | 0.3911 | **0.4601** | 0.4016 | 0.3683 | **0.5000** |

Sulla federazione vera, con la configurazione consegnata (fedprox mu 0.8), il divario si distribuisce così:

| classe | perdita | quota del divario |
|---|---|---|
| end-device | 0.0155 | 5% |
| net-device | 0.1003 | 32% |
| **server** | **0.1976** | **63%** |

La classe dei `net-device`, quella che all'apparenza sembrava il problema, **non è quella che pesa di più sul divario**, perché i server sono schiacciati da migliaia di end-device all'interno di ciascun client (nonostante i server siano presenti in 61 client di 69) e il peso di classe 3.64 non basta a farli notare. Prestare server però non risolve.

| classe | client che ne possiedono | quota del client più ricco |
|---|---|---|
| end-device | 48 su 69 | 24% |
| net-device | 42 su 69 | **56%** |
| server | **61 su 69** | 15% |

## Che cosa sbaglia il modello

![matrice di confusione](outputs/figure/04_matrice_confusione.png)

La matrice di confusione della federazione vera con la configurazione consegnata (fedprox mu 0.8), mediata sulle 25 esecuzioni:

| reale \ predetto | end-device | net-device | server |
|---|---|---|---|
| end-device (8,136) | 7,689 | 70 | 377 |
| net-device (218) | 30 | **120** | 67 |
| server (884) | **418** | 76 | **390** |

Su 884 server veri, **418 finiscono fra gli end-device** e 390 vengono riconosciuti.

![F1 per classe](outputs/figure/03_f1_per_classe.png)

Con la configurazione consegnata la F1 degli end-device sale piano lungo tutti e cinquanta i round, da 0.91 a 0.95, senza mai assestarsi, e quella dei server fa lo stesso percorso da 0.38 a 0.47. I net-device invece oscillano in base alla partecipazione del client 5, che da solo contiene 1,021 dei 1,832 net-device della federazione; il client 0 ha 17,398 indirizzi e appena 6 net-device, e siccome l'aggregazione pesa per numero di campioni vale più del triplo del client 5. Nei round in cui il client 5 non partecipa, la media cancella quello che aveva insegnato.

## Perchè non si divide per istituzione

Un'alternativa considerata era usare l'istituzione al posto della subnet come criterio di divisione fra i
client. Contata sugli stessi indirizzi, viene fuori quasi la stessa federazione:

| | per subnet (quella in uso) | per istituzione |
|---|---|---|
| client dopo la soglia di 10 indirizzi | 69 | 41 |
| client più grande | 17,398 indirizzi | 17,398 indirizzi |
| sua quota sul totale | 21.1% | 21.0% |
| net-device: quanti ne ha chi ne ha di più | 55.7% | 54.0% |

Lo stesso gigante da 17,398 indirizzi, la stessa quota di un quinto della federazione, la stessa classe minoritaria concentrata in un partecipante solo.
**Le 69 subnet che diventano client appartengono a sole 38 istituzioni, e per 31 di quelle 38 l'istituzione contiene esattamente una subnet.** Per otto casi su dieci istituzione e subnet sono lo stesso raggruppamento chiamato in due modi.

Forzare i 69 client spezzando le istituzioni fra più client è stato scartato perchè non corrisponderebbe a nessuna entità che esiste davvero.

## Come si risale alla configurazione di un risultato

Gli shard vengono sovrascritti a ogni rigenerazione, quindi in `shards/` c'è sempre e solo l'ultima partizione usata. Perchè un risultato di due settimane fa resti interpretabile, la modalità viene registrata in tre punti:

1. in `shards/meta.json`, nei campi `partition`, `rebalance_net_device` e `mini_dataset`
2. nel nome del file di storico, per esempio `history_fedavg_random_ft0.5_lrd0.97_69c_50r_*.json`, con `_mu0.8` in più quando la strategia è FedProx e `-md500` dopo la partizione quando c'è l'insieme condiviso
3. dentro ogni storico, nel blocco `partition`

`tools/grafici.py` usa partizione, ribilanciamento, insieme condiviso, passo del server di FedAdam e FedYogi e `proximal-mu` di FedProx come parte della chiave con cui raggruppa le run, così configurazioni diverse non finiscono mai mediate insieme.

## Conclusione

| | macro-F1 |
|---|---|
| RNN centralizzata baseline | 0.7355 |
| RNN federata, migliore configurazione | **0.6310** |

La configurazione migliore è **FedProx con proximal-mu 0.8, metà dei client per round, decadimento 0.97, cinquanta round**.

Il costo del federated learning su questo problema è di **0.1045** sulla macro-f1, e si distribuisce poco più di sei decimi sui `server`, circa un terzo sui `net-device` e quasi niente sugli end-device.

Conclusioni elencate:

1. **Il divario dipende da come le classi sono distribuite, non da quanto sono grandi i client.** Sparpagliando le classi si recupera il 53% del divario di FedAvg mentre pareggiando anche le dimensioni si peggiora.
2. **I pesi di classe non bastano, e prestare dati fra i client non aiuta.** Il peso 15 assegnato ai net-device funziona dentro un client che i net-device ce li ha, e portarli a dieci in tutti i client non cambia niente. Dare a tutti lo stesso insieme bilanciato fa addirittura scendere il consegnato a 0.5904, perché quell'insieme domina i conteggi aggregati e schiaccia i pesi di classe da 15 a 1.58. Il problema non è quanti esemplari ha ciascun client, è che dentro ogni client le proporzioni fra le classi non somigliano a quelle globali.
3. **La federazione reale è imprevedibile, non solo peggiore.** FedAvg consegna fra 0.52 e 0.65 a seconda della run, contro un intervallo di due centesimi con i dati mescolati. Con FedProx e mu=0.8 invece l'intervallo si stringe a 0.59-0.66 e la deviazione standard scende da 0.0299 a 0.0215.
4. **Trattenere i client vicino al modello globale è la strategia più efficace fra quelle provate, mentre adattare il passo lato server non serve.** FedAdam e FedYogi stanno entrambe sotto FedAvg e sono indistinguibili fra loro. Portare `proximal-mu` da 0.1 a 0.8 vale +0.0275 sul consegnato e sposta i `server` di poco più di quattro centesimi, che è la classe su cui nessun altro intervento aveva ottenuto niente. È arrivata dopo le prime 250 esecuzioni.
