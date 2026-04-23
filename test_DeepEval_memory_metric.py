import os
import json
import asyncio
import time
import pytest
import pandas as pd
from dotenv import load_dotenv

# ==========================================
# 1. SETUP AMBIENTE E TIMEOUT
# ==========================================
os.environ["CONFIDENT_METRIC_LOGGING_VERBOSE"] = "0"
os.environ["DEEPEVAL_DISABLE_METRIC_LOGGING"] = "YES"
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"
os.environ["CONFIDENT_API_KEY"] = "None"
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "300"
os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = "600"

# Import specifici per la memoria
from deepeval.test_case import ConversationalTestCase, Turn
from deepeval.metrics import KnowledgeRetentionMetric

load_dotenv("api_key.env")
os.environ["OPENAI_API_KEY"] = os.environ.get("API_KEY")

CSV_FILE = "Risultati_Memoria_Mimir.csv"

# Pulizia iniziale
if os.path.exists(CSV_FILE):
    os.remove(CSV_FILE)

# Configurazione Metrica Memoria (con gpt-5-nano o il modello che preferisci)
# Nota: La metrica della memoria richiede un modello intelligente come giudice
memory_metric = KnowledgeRetentionMetric(threshold=0.7, model="gpt-5-nano", async_mode=True)


@pytest.mark.parametrize("conv", json.load(open("conversazioni_memoria.json", "r", encoding="utf-8")))
def test_mimir_memory(conv):
    # 1. TRASFORMAZIONE: Usiamo i nomi corretti per la classe Turn
    turns = []
    for t in conv["turni"]:
        # Aggiungiamo il messaggio dell'utente
        turns.append(Turn(role="user", content=t["input"]))
        # Aggiungiamo la risposta dell'assistente
        turns.append(Turn(role="assistant", content=t["actual_output"]))

    # 2. CREAZIONE DEL CASO CONVERSAZIONALE
    convo_test_case = ConversationalTestCase(turns=turns)

    # 3. ESECUZIONE ASINCRONA
    start_time = time.perf_counter()

    async def run_eval():
        await memory_metric.a_measure(convo_test_case)

    asyncio.run(run_eval())

    duration = round(time.perf_counter() - start_time, 2)

    # 4. RACCOLTA DATI
    status = "PASSED" if memory_metric.score >= memory_metric.threshold else "FAILED"

    result = {
        "Trace_ID": conv["id_traccia_finale"],
        "Num_Turni": conv["numero_turni"],
        "Punteggio_Memoria": memory_metric.score,
        "Soglia": memory_metric.threshold,
        "Esito": status,
        "Durata_s": duration,
        "Ragionamento": str(memory_metric.reason).replace("\n", " ").replace(";", ",")
    }

    # 5. SALVATAGGIO CSV (Con il tuo sistema di retry corazzato)
    df = pd.DataFrame([result])
    for attempt in range(10):
        try:
            header = not os.path.exists(CSV_FILE)
            df.to_csv(CSV_FILE, mode='a', index=False, sep=";", encoding="utf-8-sig", header=header)
            break
        except Exception:
            time.sleep(0.5)