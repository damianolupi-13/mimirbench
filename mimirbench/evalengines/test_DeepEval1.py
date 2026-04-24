import os
import json
import asyncio
import time
import pytest
import pandas as pd
from dotenv import load_dotenv

# 1. SETUP AMBIENTE E TIMEOUT (TUTTI inclusi)
os.environ["CONFIDENT_METRIC_LOGGING_VERBOSE"] = "0"
os.environ["DEEPEVAL_DISABLE_METRIC_LOGGING"] = "YES"
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"
os.environ["CONFIDENT_API_KEY"] = "None"

# I tuoi timeout fondamentali
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "300"
os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = "600"

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    GEval
)

load_dotenv("../../api_key.env")
os.environ["OPENAI_API_KEY"] = os.environ.get("API_KEY")

CSV_FILE = "Risultati_Mimir_Pytest.csv"

# PULIZIA AUTOMATICA (Funziona come hai confermato tu)
if os.path.exists(CSV_FILE):
    try:
        os.remove(CSV_FILE)
        print(f"\n🧹 [PULIZIA] File {CSV_FILE} rimosso.")
    except Exception:
        pass

# 2. DEFINIZIONE METRICHE GLOBALI
faithfulness = FaithfulnessMetric(threshold=0.7, model="gpt-5-nano", async_mode=True)
relevancy = AnswerRelevancyMetric(threshold=0.7, model="gpt-5-nano", async_mode=True)
context_relevancy = ContextualRelevancyMetric(threshold=0.4, model="gpt-5-nano", async_mode=True)
"""professional_tone = GEval(
    name="Professional Tone",
    criteria="Determina se la risposta è scritta in un tono aziendale e professionale.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-5-nano",
    threshold=0.8,
    async_mode=True
)"""

lista_metriche = [faithfulness, relevancy, context_relevancy]# professional_tone]


@pytest.mark.parametrize("item", json.load(open("dati_valutazione.json", "r", encoding="utf-8")))
def test_mimir_chatbot(item):
    test_case = LLMTestCase(
        input=item["input"],
        actual_output=item["actual_output"],
        retrieval_context=item["retrieval_context"]
    )

    # ESECUZIONE PARALLELA
    # --- CRONOMETRO START ---
    start_time = time.perf_counter()

    async def run_metrics():
        await asyncio.gather(*[m.a_measure(test_case) for m in lista_metriche])

    asyncio.run(run_metrics())

    # --- CRONOMETRO END ---
    end_time = time.perf_counter()
    duration = round(end_time - start_time, 2)  # Tempo impiegato per le 4 metriche

    # 3. RACCOLTA RISULTATI (Locale per questo test)
    results_to_save = []
    for metric in lista_metriche:
        real_status = "PASSED" if metric.score >= metric.threshold else "FAILED"

        results_to_save.append({
            "Input": item["input"][:100],
            "Metrica": metric.__class__.__name__,
            "Punteggio": metric.score,
            "Soglia": metric.threshold,
            "Esito": real_status,
            "Durata (s)": duration,
            "Motivazione": str(metric.reason).replace("\n", " ").replace(";", ",")
        })

    # 4. SALVATAGGIO SICURO (FUORI DAL FOR DELLE METRICHE!)
    df = pd.DataFrame(results_to_save)

    # Sistema di retry senza librerie esterne
    for attempt in range(20):
        try:
            header = not os.path.exists(CSV_FILE)
            df.to_csv(CSV_FILE, mode='a', index=False, sep=";", encoding="utf-8-sig", header=header)
            break  # Se scrive, esce dal ciclo
        except Exception:
            # Se il file è bloccato da un altro worker, aspetta un istante variabile
            time.sleep(0.5 + (attempt * 0.1))