#Primo test con API nuova

import os
import json
import uuid
import time
import requests
import pandas as pd
from dotenv import load_dotenv

# Caricamento variabili d'ambiente
load_dotenv("test_chat_key.env")
CHAT_API_KEY = os.environ.get("CHAT_API_KEY")

if not CHAT_API_KEY:
    raise ValueError("Variabile d'ambiente CHAT_API_KEY non definita.")

# Parametri di configurazione API
API_URL = "https://chatbot-tesi-lupi-production.up.railway.app/api/chat"
HEADERS = {
    "Authorization": f"Bearer {CHAT_API_KEY}",
    "Content-Type": "application/json"
}


def esegui_test_conversazionale(input_csv_path: str, output_json_path: str):
    """
    Esegue il parsing sequenziale di un dataset di test, iterando chiamate API
    con mantenimento dello stato (thread_id) e serializzazione dell'output in JSON.
    """

    # 1. Lettura del dataset generato da Ragas
    try:
        df_testset = pd.read_csv(input_csv_path)
        # Si assume che la colonna delle domande generate si chiami 'user_input'
        if 'user_input' not in df_testset.columns:
            raise KeyError("Il file CSV non contiene la colonna 'user_input'.")
        domande = df_testset['user_input'].tolist()
    except Exception as e:
        raise RuntimeError(f"Errore durante l'acquisizione del dataset CSV: {e}")

    # 2. Inizializzazione della sessione di test
    thread_id = f"test-session-{uuid.uuid4().hex[:8]}" #definito fuori per avere le domande tutte in una stessa conversazione
    test_id_nonce = str(uuid.uuid4())
    storico_conversazione = []

    print(f"Avvio esecuzione test. Thread ID assegnato: {thread_id}")
    print(f"Totale iterazioni previste: {len(domande)}\n")

    # 3. Iterazione e invio payload
    for indice, domanda in enumerate(domande):
        payload = {
            "thread_id": thread_id,
            "message": domanda,
            "tags": ["env:test"],
            "metadata": {"test_id": test_id_nonce}
        }

        start_time = time.perf_counter()

        try:
            response = requests.post(API_URL, json=payload, headers=HEADERS)
            response.raise_for_status()  # Solleva eccezione per codici HTTP 4xx/5xx

            dati_risposta = response.json()
            risposta_testo = dati_risposta.get("text", "")

        except requests.exceptions.RequestException as req_err:
            print(f"Iterazione {indice + 1} fallita - Errore di rete/API: {req_err}")
            risposta_testo = f"[ERRORE API: {req_err}]"

        latenza_sec = round(time.perf_counter() - start_time, 3)

        # Costruzione del nodo strutturato per il turno corrente
        turno_data = {
            "turno": indice + 1,
            "input": domanda,
            "actual_output": risposta_testo,
            "latenza_chiamata_s": latenza_sec
        }

        storico_conversazione.append(turno_data)

        # Output console a scopo di monitoraggio dell'esecuzione
        print(f"[{indice + 1}/{len(domande)}] Latenza: {latenza_sec}s")

        # Applicazione di un delay opzionale per prevenire il rate-limiting lato server
        time.sleep(1.0)

        # 4. Formattazione e serializzazione nel nodo radice
    output_strutturato = [{
        "id_traccia_finale": thread_id,
        "numero_turni": len(storico_conversazione),
        "turni": storico_conversazione
    }]

    # 5. Scrittura del buffer su file system
    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(output_strutturato, json_file, indent=4, ensure_ascii=False)

    print(f"\nEsecuzione terminata. Storico serializzato in: {output_json_path}")


# Blocco di esecuzione principale
if __name__ == "__main__":
    PATH_INPUT_CSV = "output/testset_test3.csv"  # Da adattare al path effettivo
    PATH_OUTPUT_JSON = "output/storico_api_test3.json"

    esegui_test_conversazionale(PATH_INPUT_CSV, PATH_OUTPUT_JSON)
