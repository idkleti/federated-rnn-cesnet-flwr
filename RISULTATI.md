# RISULTATI DEGLI ESPERIMENTI
Il file usa la convenzione numerica americana (1,234.56).

Sono state effettuate **250 esecuzioni complete** su WSL2 (16 core, 7.4 GB di RAM, RTX 5060 con 8 GB), per un totale di circa 28 ore di solo addestramento.
I comandi che le hanno prodotte stanno in [`tools/esperimenti.sh`](tools/esperimenti.sh), gli storici round per round in `outputs/history_*.json`, e le figure si rigenerano con `python tools/grafici.py`.

**Ogni configurazione è stata eseguita 25 volte.**

## Confronti

Il riferimento è la RNN centralizzata di base fornita, che ottiene **macro-F1 0.7615** sul test set.

Il confronto viene effettuato sullo stesso test set poichè lo split avviene con lo stesso seed (111) e le stesse proporzioni, ottenendo gli stessi conteggi per classe (8,136 end-device, 218 net-device e 884 server).

Il modello inviato al server viene selezionato in base alla macro-f1 score migliore della run sui dati di validazione dei client, senza mai guardare il test. Si utilizza come metrica la macro-f1 score rispetto all'accuratezza poichè bisogna tenere conto dello sbilanciamento delle classi nel dataset.

Il test set del server è lo stesso identico in tutte e quattro le partizioni confrontate più sotto. La
divisione di holdout avviene prima di distribuire gli indirizzi ai client, quindi non dipende da come poi
vengono divisi, e `prepare_data.py` se lo ritrova già scritto per poi riconoscerlo dagli identificatori e tenerselo.

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
* **round**: numero di round effettuati nell'addestramento
* **consegnato**: macro-f1 sul test del modello selezionato
* **stab.**: quanto oscilla la macro-f1 negli ultimi 31 round
* **net-dev sotto 0.01**: in quanti degli ultimi 31 round la F1 dei net-device era praticamente azzerata

In tutti gli esperimenti si hanno 2 epoche locali, Adam a 6.92e-4, weight decay 0.01, batch 64.

| strategia | ft | lrd | round | consegnato | stab. | net-dev sotto 0.01 |
|---|---|---|---|---|---|---|
| **fedavg** | 0.5 | 0.97 | 50 | **0.6101**  | 0.0575 | 13/31 |
| fedprox | 0.5 | 0.97 | 50 | 0.6035  | 0.0501 | 11/31 |
| fedavg | 0.5 | — | 100 | 0.5983  | 0.0570 | 14/31 |
| fedavg | 0.3 | — | 50 | 0.5911  | 0.0789 | 14/31 |
| fedavg | 0.5 | — | 50 | 0.5910  | 0.0577 | 12/31 |
| fedadam | 0.5 | 0.97 | 50 | 0.5856 | 0.0587 | 9/31 |
| fedavg | 1.0 | — | 50 | 0.5249 | **0.0175** | **2/31** |
| *centralizzata* | | | | *0.7615* | | |

![confronto fra le configurazioni](outputs/figure/02_confronto_strategie.png)

![curve di apprendimento](outputs/figure/01_curve_macro_f1.png)

## Le strategie di aggregazione sono quasi equivalenti

A parità di partecipazione e decadimento, le tre strategie stanno in poco più di due centesimi:

| | consegnato | dev std |
|---|---|---|
| fedavg | 0.6101 | ± 0.0299 |
| fedprox | 0.6035 | ± 0.0421 |
| fedadam | 0.5856 | ± 0.0417 |

FedAvg e FedProx sono praticamente indistinguibili. FedAvg viene scelto per la deviazione standard minore, cioè perchè si ripete più uguale a sé stesso.


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

Con la partecipazione totale invece il calo vale 0.066 nonostante sia la configurazione più stabile di tutte; infatti, si ha deviazione standard 0.0175 contro 0.05-0.08 delle altre, e solo 2 round su 31 con i net-device azzerati contro i 9-14 degli altri. Da qui ricaviamo che un addestramento "instabile" produce dei round con macro-f1 score elevata fra cui scegliere, mentre uno stabile resta piatto e non offre niente di meglio.

## Quanto conta la forma della federazione

In questa sezione, la configurazione utilizzata è **FedAvg, fraction-train = 0.5, lr-decay 0.97 e 50 round**, l'unica cosa che cambia è la divisione degli indirizzi fra i client.
Gli indirizzi sono sempre 82,527 e i client sempre 69.

| | consegnato | min-max | divario |
|---|---|---|---|
| subnet, la federazione vera | 0.6101 ± 0.0299 | 0.5174-0.6515 | 0.1514 |
| subnet + 425 net-device condivisi | 0.6141 ± 0.0227 | 0.5366-0.6499 | 0.1474 |
| **casuale a dimensioni reali** | **0.6767 ± 0.0069** | 0.6609-0.6928 | **0.0848** |
| casuale a blocchi uguali | 0.6440 ± 0.0058 | 0.6325-0.6526 | 0.1175 |

Un paio di note sulla suddivisione casuale (usano gli stessi indirizzi e numero di client):
* **casuale a blocchi uguali**: gli indirizzi vengono mescolati e divisi in 69 pezzi identici
* **casuale a dimensioni reali**: mescolati allo stesso modo, ma le dimensioni restano quelle delle subnet vere, quindi resta il client da 17,398 indirizzi e restano quelli da 10

| | subnet | casuale | casuale a dimensioni reali |
|---|---|---|---|
| indirizzi per client | 10 / 34 / 17,398 | tutti ~1,196 | 10 / 34 / 17,398 |
| client con almeno un net-device | 42 su 69 | 69 su 69 | 43 su 69 |
| net-device di chi ne ha di più | 55.7% | 2.1% | 19.2% |

Guardando la seconda riga si può notare come, mantenendo le dimensioni vere ma mescolando i dati, **il fatto che metà federazione non abbia nemmeno un net-device dipende da quanto sono piccoli quei client e non da come sono fatte le subnet**.

La concentrazione del 55.7% invece crolla al 19.2%.

La federazione reale produce un risultato più basso e imprevedibile a seconda di come gira.

## Le classi contano, le dimensioni no

| passaggio | cosa cambia | effetto |
|---|---|---|
| subnet → casuale a dimensioni reali | si sparpagliano le **classi**, le dimensioni restano quelle vere | **+0.0666** (10.8 volte l'errore) |
| casuale a dimensioni reali → casuale a blocchi uguali | si pareggiano anche le **dimensioni** | **−0.0327** (18.1 volte l'errore) |

La prima divisione dei dati migliora il risultato mentre la seconda divisione peggiora il risultato totale, quindi una federazione di 69 client tutti uguali va peggio di una in cui pochi client sono enormi e molti minuscoli.

La spiegazione plausibile è che con pochi client grossi la media pesata di FedAvg sia dominata da aggiornamenti calcolati su molti dati, mentre 34 client da 1,200 indirizzi producono 34 aggiornamenti rumorosi da mediare fra loro.

Quindi dei **0.1514** di divario rispetto al modello centralizzato:
* **circa il 44% (0.0666)** dipende da come le classi sono distribuite fra le subnet
* il restante **56% (0.0848)** è il costo di federare in sé, che resta anche con la partizione migliore

## Il ribilanciamento della classe rara non funziona

L'esperimento consisteva nel distribuire copie di net-device ai client che ne avevano pochi, fino a portarli tutti ad almeno dieci e senza duplicarle. Servono 425 serie, cioè lo 0.5% del pool, e i client senza nemmeno un net-device passano da 27 a zero. Con questa prova andiamo a **rilassare l'ipotesi del federated learning**, perchè delle serie passano da un client a un altro.

Il guadagno è **+0.0039** e la F1 dei net-device passa da 0.4969 a **0.4976**, che vuol dire poco.

L'idea viene da Zhao et al. 2018, *Federated Learning with Non-IID Data*, dove un piccolo insieme condiviso viene distribuito a tutti i partecipanti e si misura quanto si guadagna al crescere della frazione condivisa.

C'è da notare però come da loro l'insieme condiviso sia un dataset a parte che gli autori dichiarano non sensibile proprio perché non appartiene a nessun client (sta sul server e viene distribuito a tutti in fase di inizializzazione), mentre nel progetto le serie sono traffico reale di alcune istituzioni che finisce dentro altre istituzioni, quindi il prezzo in riservatezza è di natura diversa e va dichiarato.

Alcune note sulla prova:

* le 425 serie prestate sono **tutte diverse**, nessuna viene duplicata. Ogni serie finisce in due client soltanto, il suo proprietario e un ricevente, quindi il modello non può impararne a memoria poche ripetute ovunque
* i donatori restano sempre sopra la soglia, quindi nessuno viene impoverito per arricchire un altro
* le righe prestate stanno in coda allo shard e **non entrano mai nella validazione locale**. Se ci entrassero, il modello verrebbe valutato su una serie già vista in addestramento a casa di un altro client, e siccome sono tutte net-device il punteggio salirebbe proprio sulla classe che l'esperimento vuole misurare

## Dove si perde il divario, classe per classe

F1 per classe:

| classe | centralizzata | subnet | + net-device | dimensioni reali |
|---|---|---|---|---|
| end-device | 0.9642 | 0.9423 | 0.9430 | 0.9449 |
| net-device | 0.6304 | 0.4969 | 0.4976 | **0.5851** |
| server | 0.6900 | 0.3911 | 0.4016 | **0.5000** |

Sulla federazione vera il divario si distribuisce così:

| classe | perdita | quota del divario |
|---|---|---|
| end-device | 0.0219 | 5% |
| net-device | 0.1335 | 29% |
| **server** | **0.2989** | **66%** |

Due terzi del divario vengono dalla classe `server`, che passa da 0.3911 a 0.4016 aggiungendo i net-device, ma diventa 0.5000 sparpagliando tutte le classi. La classe dei `net-device`, quella che all'apparenza sembrava il problema, era la **classe sbagliata da prestare** poiché i server sono schiacciati da migliaia di end-device all'interno di ciascun client (nonostante i server siano presenti in 61 client di 69) e il peso di classe 3.64 non basta a farli notare.

| classe | client che ne possiedono | quota del client più ricco |
|---|---|---|
| end-device | 48 su 69 | 24% |
| net-device | 42 su 69 | **56%** |
| server | **61 su 69** | 15% |

## Che cosa sbaglia il modello

![matrice di confusione](outputs/figure/04_matrice_confusione.png)

La matrice di confusione della federazione vera, mediata sulle 25 esecuzioni:

| reale \ predetto | end-device | net-device | server |
|---|---|---|---|
| end-device (8,136) | 7,676 | 97 | 363 |
| net-device (218) | 27 | **148** | 42 |
| server (884) | **446** | 124 | 313 |

Su 884 server veri, **446 finiscono fra gli end-device** e solo 313 vengono riconosciuti. I net-device invece se la cavano meglio di quanto ci si aspettasse.

![F1 per classe](outputs/figure/03_f1_per_classe.png)

La F1 degli end-device sale nei primi round e poi resta piatta sopra 0.93. Quella dei net-device oscilla in base alla partecipazione del client 5, che contiene 1,021 dei 1,832 net-device totali; il client 0 invece ha 17,398 indirizzi e appena 6 net-device, e siccome FedAvg pesa per numero di campioni vale da solo più del triplo del client 5. Nei round in cui il client 5 non partecipa, la media cancella quello che aveva insegnato.

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

1. in `shards/meta.json`, nei campi `partition` e `rebalance_net_device`
2. nel nome del file di storico, per esempio `history_fedavg_random_ft0.5_lrd0.97_69c_50r_*.json`
3. dentro ogni storico, nel blocco `partition`

`tools/grafici.py` usa partizione e ribilanciamento come parte della chiave con cui raggruppa le run, così configurazioni diverse non finiscono mai mediate insieme.

## Conclusione

| | macro-F1 |
|---|---|
| RNN centralizzata baseline | 0.7615 |
| RNN federata, migliore configurazione | **0.6101** |
| divario | **0.151** |

La configurazione migliore è **FedAvg, metà dei client per round, decadimento 0.97, cinquanta round**.

Il costo del federated learning su questo problema è di **0.151** sulla macro-f1, e si distribuisce due terzi sui `server`, poco meno di un terzo sui `net-device` e quasi niente sugli end-device.

Conclusioni elencate:

1. **Il divario dipende da come le classi sono distribuite, non da quanto sono grandi i client.** Sparpagliando le classi si recupera il 44% del divario; pareggiando anche le dimensioni si peggiora.
2. **I pesi di classe non bastano, ma nemmeno aggiungere dati della classe rara.** Il peso 15 assegnato ai net-device funziona dentro un client che i net-device ce li ha, e portarli a dieci in tutti i client non cambia niente. Il problema non è quanti ne ha ciascun client, è che dentro ogni client le proporzioni fra le classi non somigliano a quelle globali.
3. **La federazione reale è imprevedibile, non solo peggiore.** Consegna fra 0.52 e 0.65 a seconda della run, contro un intervallo di due centesimi con i dati mescolati.
