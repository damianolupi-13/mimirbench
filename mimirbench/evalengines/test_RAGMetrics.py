import os
import json
import asyncio
import time
import pytest
import pandas as pd

# Lettura dinamica dei path e dei tool dal JSON di configurazione
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
        case "google" | "gemini":
            os.environ["GOOGLE_API_KEY"] = chiave_salvata
        case _:
            print(f"[ATTENZIONE] Provider '{provider}' non riconosciuto. Chiave non impostata.")


from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric
)

CSV_FILE = config_data.get("MIMIR_CSV_PATH", "Risultati_Agent_Fallback.csv")
JSON_PATH = config_data.get("MIMIR_JSON_PATH", "")

def carica_dati_test(filepath):
    """Carica i dati in modo sicuro per il parametrize di Pytest."""
    if not filepath or not os.path.exists(filepath):
        print(f"\n[ATTENZIONE] Pytest non ha trovato il file JSON al percorso: {filepath}")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# Pulizia iniziale automatica
if os.path.exists(CSV_FILE):
    try:
        os.remove(CSV_FILE)
    except Exception:
        pass

# DEFINIZIONE METRICHE
faithfulness = FaithfulnessMetric(threshold=0.7, model=custom_model, async_mode=True)
relevancy = AnswerRelevancyMetric(threshold=0.7, model=custom_model, async_mode=True)
context_relevancy = ContextualRelevancyMetric(threshold=0.4, model=custom_model, async_mode=True)
"""professional_tone = GEval(
    name="Professional Tone",
    criteria="Determina se la risposta è scritta in un tono aziendale e professionale.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-5-nano",
    threshold=0.8,
    async_mode=True
)"""

lista_metriche = [faithfulness, relevancy, context_relevancy]  # professional_tone]

@pytest.mark.parametrize("item", carica_dati_test(JSON_PATH))
def test_mimir_chatbot(item):
    test_case = LLMTestCase(
        input=item["input"],
        actual_output=item["actual_output"],
        retrieval_context=item["retrieval_context"]
    )

    # ESECUZIONE PARALLELA
    start_time = time.perf_counter()

    async def run_metrics():
        await asyncio.gather(*[m.a_measure(test_case) for m in lista_metriche])

    asyncio.run(run_metrics())

    end_time = time.perf_counter()
    duration = round(end_time - start_time, 2)  # Tempo impiegato per le 4 metriche

    # RACCOLTA RISULTATI (Locale per questo test)
    results_to_save = []
    for metric in lista_metriche:
        real_status = "PASSED" if metric.score >= metric.threshold else "FAILED"

        results_to_save.append({
            "Trace_ID": item.get("id_traccia", ""),
            "Input": item["input"][:100],
            "Metrica": metric.__class__.__name__,
            "Punteggio": metric.score,
            "Soglia": metric.threshold,
            "Esito": real_status,
            "Durata (s)": duration,
            "Motivazione": str(metric.reason).replace("\n", " ").replace(";", ",")
        })

    # SALVATAGGIO
    df = pd.DataFrame(results_to_save)

    for attempt in range(20):
        try:
            header = not os.path.exists(CSV_FILE)
            df.to_csv(CSV_FILE, mode='a', index=False, sep=";", encoding="utf-8-sig", header=header)
            break  # Se scrive, esce dal ciclo
        except Exception:
            # Se il file è bloccato da un altro worker, aspetta un istante variabile
            time.sleep(0.5 + (attempt * 0.1))