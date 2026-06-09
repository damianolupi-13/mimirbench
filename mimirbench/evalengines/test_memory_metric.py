import os
import json
import asyncio
import time
import pytest
import pandas as pd

config_path = os.path.join(os.path.dirname(__file__), "mimir_config.json")
config_data = {}

if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config_data.update(json.load(f))

# Estraiamo provider e chiave dal file di configurazione
provider = config_data.get("MIMIR_EVAL_PROVIDER", "openai").lower()
chiave_salvata = config_data.get("MIMIR_EVAL_API_KEY", "")
custom_model = config_data.get("MIMIR_EVAL_MODEL", "gpt-5-nano")

# Smistamento dinamico della chiave API
if chiave_salvata:
    match provider:
        case "openai":
            os.environ["OPENAI_API_KEY"] = chiave_salvata
        case "anthropic":
            os.environ["ANTHROPIC_API_KEY"] = chiave_salvata
        case "mistral":
            os.environ["MISTRAL_API_KEY"] = chiave_salvata
        case "google" | "gemini":
            os.environ["GOOGLE_API_KEY"] = chiave_salvata
        case "cohere":
            os.environ["COHERE_API_KEY"] = chiave_salvata
        case _:
            print(f"[ATTENZIONE] Provider '{provider}' non riconosciuto. Chiave non impostata.")

from deepeval.test_case import ConversationalTestCase, Turn
from deepeval.metrics import KnowledgeRetentionMetric

CSV_FILE = config_data.get("MIMIR_CSV_PATH", "Risultati_Memoria_Fallback.csv")
JSON_PATH = config_data.get("MIMIR_JSON_PATH", "")

def carica_dati_test(filepath):
    """Carica i dati in modo sicuro per il parametrize di Pytest."""
    if not filepath or not os.path.exists(filepath):
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# Pulizia iniziale automatica
if os.path.exists(CSV_FILE):
    try:
        os.remove(CSV_FILE)
    except Exception:
        pass


# Nota: La metrica della memoria richiede un modello intelligente come giudice
memory_metric = KnowledgeRetentionMetric(threshold=0.7, model=custom_model, async_mode=True)


@pytest.mark.parametrize("conv", carica_dati_test(JSON_PATH))
def test_mimir_memory(conv):
    # Usiamo i nomi corretti per la classe Turn
    turns = []
    for t in conv["turni"]:
        # Aggiungiamo il messaggio dell'utente
        turns.append(Turn(role="user", content=t["input"]))
        # Aggiungiamo la risposta dell'assistente
        turns.append(Turn(role="assistant", content=t["actual_output"]))

    # CREAZIONE DEL CASO CONVERSAZIONALE
    convo_test_case = ConversationalTestCase(turns=turns)

    # ESECUZIONE PARALLELA
    start_time = time.perf_counter()

    async def run_eval():
        await memory_metric.a_measure(convo_test_case)

    asyncio.run(run_eval())

    duration = round(time.perf_counter() - start_time, 2)

    # RACCOLTA DATI
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

    # SALVATAGGIO
    df = pd.DataFrame([result])
    for attempt in range(10):
        try:
            header = not os.path.exists(CSV_FILE)
            df.to_csv(CSV_FILE, mode='a', index=False, sep=";", encoding="utf-8-sig", header=header)
            break
        except Exception:
            time.sleep(0.5)