import os
import sys
import json
import uuid
import time
import requests
import httpx
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from langfuse import Langfuse

# Configurazione del path per includere la root del progetto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mimirbench.testsets.contextual import ContextualTestset
from mimirbench.tracings import ContextualTraceExtractor
from mimirbench.evalengines.engines import RagEvalEngine
from mimirbench.printers.pdf_printer import MimirPDFPrinter
from mimirbench.printers.langfuse_printer import LangfusePrinter


# =====================================================================
# FUNZIONE DI SUPPORTO ALL'ESECUZIONE: PAUSA INTERATTIVA
# =====================================================================
def chiedi_continuazione(nome_prossima_fase):
    """
    Blocca l'esecuzione e chiede all'utente se procedere.
    Garantisce un'ampia spaziatura visiva nel terminale.
    """
    print("\n\n" + "=" * 60)
    print("PAUSA DI ESECUZIONE")
    print("=" * 60)
    scelta = input(f"Vuoi procedere con la {nome_prossima_fase}? (y/n): ").strip().lower()
    print("=" * 60 + "\n\n")

    if scelta not in ['s', 'si', 'sì', 'y', 'yes']:
        print("Esecuzione interrotta dall'utente.")
        sys.exit(0)


# =====================================================================
# INIZIALIZZAZIONE AMBIENTE E VARIABILI
# =====================================================================

CARTELLA_BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CARTELLA_BASE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Caricamento delle chiavi dai rispettivi file .env
load_dotenv(os.path.join(CARTELLA_BASE, "api_key.env"))
load_dotenv(os.path.join(CARTELLA_BASE, "test_chat_key.env"))
load_dotenv(os.path.join(CARTELLA_BASE, "langfusekeys.env"))
langfuse_client = Langfuse()

# Configurazione API Chatbot esterno
CHAT_API_KEY = os.environ.get("CHAT_API_KEY")
API_URL = "https://chatbot-tesi-lupi-production.up.railway.app/api/chat"
if not CHAT_API_KEY:
    raise ValueError("ERRORE: CHAT_API_KEY non configurata nel file environment.")

HEADERS = {
    "Authorization": f"Bearer {CHAT_API_KEY}",
    "Content-Type": "application/json"
}

print("=== MIMIRBENCH TASK SPECIFIC PIPELINE RUNNER ===")

# =====================================================================
# GENERAZIONE O SELEZIONE DEL TESTSET
# =====================================================================
path_testset_csv = os.path.join(OUTPUT_DIR, "testset-prova-tesi1.csv")

print("\n[FASE 1] Configurazione del Testset di Domande")
print("1) Genera un nuovo testset sintetico con Ragas (Richiede Documento)")
print("2) Utilizza un file CSV di domande pre-costituito")
scelta_input = input("Seleziona un'opzione (1 o 2): ").strip()

if scelta_input == "1":
    print("\n--> Avvio Generazione RAGAS...")
    pdf_input = input("Inserisci il percorso del documento di contesto: ").strip()

    test_size = int(input("Numero di domande da generare atteso: ").strip())

    # Sovrascrive/imposta la chiave OpenAI per Ragas
    os.environ["OPENAI_API_KEY"] = os.environ.get("API_KEY", "")

    # Limitatore del lancio parallelo di richieste ad OpenAI per garantire la correttezza dell'operazione su numeri elevati di richieste
    limitatore_connessioni = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=25, max_keepalive_connections=25)
    )

    client = AsyncOpenAI(
        http_client=limitatore_connessioni,
        max_retries=5  # Se una richiesta fallisce, ritenta fino a 5 volte in automatico
    )
    context_prompt = "IMPORTANTISSIMO: Devi generare i nomi delle Personas, le descrizioni, gli scenari e le domande finali ESCLUSIVAMENTE in questa lingua: Italiano."

    # Istanziazione e generazione tramite modulo contextual
    test_generator = ContextualTestset(
        testset_size=test_size,
        model="gpt-5-mini",
        client=client,
        embedding="text-embedding-3-small",
        context=context_prompt,
        provider="openai",
        max_tokens=8192,
        soglia_limite=50,
        pagine_da_estrarre=50
    )
    test_generator.load(pdf_input)
    test_generator.generate_testset(path_testset_csv)
    print(f"\n--> Testset sintetizzato con successo in: {path_testset_csv}")

else:
    path_custom = input("Inserisci il percorso del file CSV esistente: ").strip()
    if not os.path.exists(path_custom):
        raise FileNotFoundError(f"File {path_custom} non trovato.")
    path_testset_csv = path_custom
    print(f"\n--> Utilizzo del dataset esistente: {path_testset_csv}")

# =====================================================================
# BLOCCO CONDIZIONALE: SALTO FASI 2 E 3
# =====================================================================
print("\n" + "=" * 60)
print("SCELTA PERCORSO ESECUZIONE")
print("=" * 60)
scelta_salto = input(
    "Hai già eseguito l'interazione e l'estrazione tracce da Langfuse e vuoi passare direttamente alla valutazione con DeepEval (FASE 4)? (y/n): ").strip().lower()

salta_interazione = False
if scelta_salto in ['s', 'si', 'sì', 'y', 'yes']:
    salta_interazione = True
    nome_file_tracce = input(
        "Inserisci il nome del file JSON delle tracce: ").strip()
    path_traces_json = os.path.join(OUTPUT_DIR, nome_file_tracce)

    if not os.path.exists(path_traces_json):
        raise FileNotFoundError(
            f"File {path_traces_json} non trovato. Assicurati di aver inserito il nome corretto e che si trovi nella cartella 'output'.")

    print(f"\n--> Salto Fasi 2 e 3. Utilizzo il file tracce esistente: {path_traces_json}")
else:
    chiedi_continuazione("FASE 2 [Interazione con l'Agente]")

# Esegue le fasi 2 e 3 solo se l'utente non ha deciso di saltarle
if not salta_interazione:
    # =====================================================================
    # INTERAZIONE CON L'AGENTE E TRACING
    # =====================================================================
    print("\n[FASE 2] Interazione con l'Agente")
    try:
        df_testset = pd.read_csv(path_testset_csv)
        colonna_domande = 'user_input' if 'user_input' in df_testset.columns else df_testset.columns[0]
        domande = df_testset[colonna_domande].tolist()
    except Exception as e:
        raise RuntimeError(f"Errore nell'acquisizione del dataset delle domande: {e}")

    # Generiamo un ID univoco globale per raggruppare questa specifica sessione di test
    test_id_sessione = str(uuid.uuid4())
    storico_completo_esperimento = []

    print(f"Identificativo univoco dell'esperimento (test_id): {test_id_sessione}")
    print(f"\nInizio interazione (Totale: {len(domande)} domande).")

    for indice, domanda in enumerate(domande):
        # NOTA: Per effettuare una corretta RAG Evaluation
        # viene generato un thread_id dedicato per ogni singola domanda.
        thread_id_singolo = f"rag-eval-{uuid.uuid4().hex[:10]}"

        payload = {
            "thread_id": thread_id_singolo,
            "message": domanda,
            "tags": ["env:test"],
            "metadata": {"test_id": test_id_sessione}  # <- Stesso ID per tutta la sessione
        }

        start_time = time.perf_counter()
        try:
            response = requests.post(API_URL, json=payload, headers=HEADERS)
            response.raise_for_status()  # Solleva eccezione per codici HTTP 4xx/5xx
            risposta_testo = response.json().get("text", "")
        except requests.exceptions.RequestException as req_err:
            # Cattura in modo specifico i problemi di comunicazione con il server
            print(f"   [!] Errore di Rete/API alla domanda {indice + 1}: {req_err}")
            risposta_testo = f"[ERRORE DI RETE: {req_err}]"

        except ValueError as val_err:
            # Cattura errori se la risposta del server non è un JSON valido
            print(f"   [!] Errore Parsing JSON alla domanda {indice + 1}: {val_err}")
            risposta_testo = "[ERRORE PARSING JSON DAL SERVER]"

        latenza_sec = round(time.perf_counter() - start_time, 3)

        storico_completo_esperimento.append({
            "turno": indice + 1,
            "input": domanda,
            "actual_output": risposta_testo,
            "latenza_chiamata_s": latenza_sec,
            "thread_id_utilizzato": thread_id_singolo
        })

        print(f"   [{indice + 1}/{len(domande)}] Inviata con successo. Latenza: {latenza_sec}s")
        time.sleep(1.5)  # Prevenzione del rate-limiting lato server

    # Salva lo storico delle interazioni locali per sicurezza e tracciabilità interna
    path_storico_json = os.path.join(OUTPUT_DIR, "storico_interazioni_agente_prova_tesi1.json")
    with open(path_storico_json, "w", encoding="utf-8") as f:
        json.dump(storico_completo_esperimento, f, indent=4, ensure_ascii=False)

    print(f"\n--> Storico dell'interazione salvato con successo in: {path_storico_json}")

    chiedi_continuazione("FASE 3 [Sincronizzazione e Download Tracce]")

    # =====================================================================
    # LIVELLO DI ESTRAZIONE TELEMETRICA
    # =====================================================================
    print("\n[FASE 3] Sincronizzazione Cloud e Download delle Tracce")
    # Tempo di attesa: diamo tempo ai server di Langfuse di ricevere e indicizzare i log asincroni dell'agente
    tempo_attesa_cloud = 15
    print(f"--> Attivazione tempo di tolleranza ({tempo_attesa_cloud}s) per l'aggiornamento di Langfuse in cloud...")
    time.sleep(tempo_attesa_cloud)

    print("\n--> Avvio estrazione delle tracce storicizzate...")
    extractor = ContextualTraceExtractor(langfuse_client)
    path_traces_json = os.path.join(OUTPUT_DIR, "extracted_traces_data_prova_tesi1.json")

    # L'extractor usa il test_id univoco che raggruppa le domande appena generate
    extractor.extracting(path_traces_json, test_id_sessione)
    print(f"\n--> Download completato. Dataset telemetrico salvato in: {path_traces_json}")

chiedi_continuazione("FASE 4 [Motore di Valutazione]")

# =====================================================================
# MOTORE DI VALUTAZIONE AUTOMATIZZATO (EVAL ENGINE)
# =====================================================================
print("\n[FASE 4] Inizializzazione ed Esecuzione dell'Engine di Valutazione")
path_risultati_csv = os.path.join(OUTPUT_DIR, "risultati_valutazione_rag_prova_tesi1.csv")

engine = RagEvalEngine(
    json_dati_path=path_traces_json,
    output_csv_path=path_risultati_csv,
    parallel_launches=2,  # Esecuzione parallela tramite Pytest-xdist
    provider="openai",
    model="gpt-5-mini"
)

print("\n--> Lancio del subprocess di collaudo con calcolo asincrono delle metriche...")
engine.run()
print(f"\n--> Valutazione conclusa. File dei risultati generato: {path_risultati_csv}")

chiedi_continuazione("FASE 5 [Generazione Output e Reportistica]")

# =====================================================================
# REPORTISTICA E OUTPUT FORMATTATO (PRINTERS SELECTION)
# =====================================================================
print("\n[FASE 5] Generazione Output e Reportistica Finale")
print("\nSeleziona la modalità di esportazione dei risultati:")
print("1) Genera esclusivamente il Report PDF matematico locale")
print("2) Effettua esclusivamente il Push dei punteggi sulla Dashboard di Langfuse")
print("3) Esegui entrambe le esportazioni (PDF + Langfuse)")
scelta_output = input("Seleziona un'opzione (1, 2 o 3): ").strip()

if scelta_output in ["1", "3"]:
    print("\n--> Generazione Report PDF...")
    path_report_pdf = os.path.join(OUTPUT_DIR, "Mimir_Evaluation_Report_prova_tesi1.pdf")
    pdf_maker = MimirPDFPrinter(csv_file_path=path_risultati_csv, output_pdf_path=path_report_pdf)
    pdf_maker.genera_report()
    print(f"--> Documento PDF pronto: {path_report_pdf}")

if scelta_output in ["2", "3"]:
    print("\n--> Sincronizzazione dei punteggi e dei ragionamenti su Langfuse Cloud...")
    langfuse_exporter = LangfusePrinter(csv_file_path=path_risultati_csv, langfuse_client=langfuse_client)
    langfuse_exporter.push_scores()

    # Tempo di attesa finale per assicurarsi il completamento del flush() di Langfuse prima della chiusura dello script
    print("--> Attesa per sincronizzazione...")
    time.sleep(3)

print("\n==========================================================")
print(" PIPELINE COMPLETATA CON SUCCESSO IN OGNI LAYER OPERATIVO")
print("==========================================================")