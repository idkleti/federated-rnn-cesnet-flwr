"""
    Classificazione del tipo di dispositivo utilizzando il dataset CESNET-TimeSeries24 con RNN federata a partire da una RNN centralizzata fornita
    con dati divisi per subnet istituzionale.

    Il package contiene:    
        - config.py         costanti e variabili di configurazione del progetto
        - task.py           modello, addestramento locale, valutazione e metriche
        - data.py           lettura shard e costruzione DataLoader
        - client_app.py     applicazione nodo client che possiede i dati
        - server_app.py     applicazione nodo server che coordina la federazione

    I dati vengono preparati e divisi per client all'interno del modulo 'prepare_data.py' perchè è un passo che viene eseguito una sola volta (per
    simulare i dati divisi per subnet) al di fuori della simulazione.
"""