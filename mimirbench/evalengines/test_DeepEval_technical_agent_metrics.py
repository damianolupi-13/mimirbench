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

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval, BaseMetric

load_dotenv("../../testdata/api_key.env")
os.environ["OPENAI_API_KEY"] = os.environ.get("API_KEY")

CSV_FILE = "Risultati_Agent_Pytest.csv"

# Pulizia automatica del file risultati precedente all'avvio dei test
if os.path.exists(CSV_FILE):
    try:
        os.remove(CSV_FILE)
        print(f"\n🧹 [PULIZIA] File {CSV_FILE} rimosso.")
    except Exception:
        pass


# ==========================================
# 2. DEFINIZIONE METRICHE CUSTOM E DETERMINISTICHE
# ==========================================

class ToolHallucinationMetric(BaseMetric):
    def __init__(self, available_tools: list):
        self.available_tools = available_tools
        self.threshold = 1.0

    # Firma corretta con *args, **kwargs e tipo di ritorno
    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        actual_tool = test_case.additional_metadata.get("tool_called", "")

        if not actual_tool or actual_tool == "Nessun tool":
            self.success = True
            self.score = 1.0
            self.reason = "Nessun tool chiamato. Comportamento testuale standard."
        elif actual_tool in self.available_tools:
            self.success = True
            self.score = 1.0
            self.reason = f"Tool valido: l'agente ha usato '{actual_tool}' che è tra quelli consentiti."
        else:
            self.success = False
            self.score = 0.0
            self.reason = f"ALLUCINAZIONE: L'agente ha tentato di usare '{actual_tool}', che NON esiste."

        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Tool Hallucination"


class LatencyMetric(BaseMetric):
    def __init__(self, max_seconds: float):
        self.max_seconds = max_seconds
        self.threshold = 1.0

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        actual_latency = test_case.additional_metadata.get("latency_seconds", 0)
        self.success = actual_latency <= self.max_seconds
        self.score = 1.0 if self.success else 0.0
        self.reason = f"Latenza: {actual_latency}s (Soglia: {self.max_seconds}s)"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Response Latency"


class TokenEfficiencyMetric(BaseMetric):
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.threshold = 1.0

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        tokens_usati = test_case.additional_metadata.get("total_tokens", 0)
        self.success = tokens_usati <= self.max_tokens
        self.score = 1.0 if self.success else 0.0
        self.reason = f"Consumo Token: {tokens_usati} (Soglia massima impostata: {self.max_tokens})"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Token Efficiency"


# ==========================================
# 3. DEFINIZIONE METRICA AGENTICA (GEval)
# ==========================================
agentic_synthesis = GEval(
    name="Agentic Synthesis",
    criteria="""Valuta la risposta finale dell'agente seguendo queste due regole rigide:
1. SE il retrieval_context CONTIENE DATI (documenti RAG o risultati di tool): L'agente deve rispondere basandosi in modo logico e fedele ESCLUSIVAMENTE su quei dati. Non deve inventare informazioni tecniche o fattuali.
2. SE il retrieval_context È VUOTO ([]): Significa che l'utente sta facendo "small talk", sta facendo domande generali, oppure sta continuando una conversazione precedente basandosi sulla memoria dell'agente. In questo caso, valuta SOLO se la risposta è sensata, educata e coerente con l'Input.
IMPORTANTE: Se l'agente menziona nomi propri (es. il nome dell'utente) o dettagli di turni precedenti, ASSUMI CHE LI ABBIA RECUPERATI CORRETTAMENTE DALLA SUA MEMORIA. Non considerarli allucinazioni e non penalizzare l'agente.
IMPORTANTE: È normale e accettabile che l'agente ricordi all'utente il proprio scopo (es. menzionare documenti o il suo nome) anche durante i saluti. Non penalizzare tali introduzioni.""",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
    model="gpt-4o",
    threshold=0.7,
    async_mode=True
)

# --- LISTA DEI TOOL REALI DI MIMIR ---
TOOL_DISPONIBILI_MIMIR = [
    "hybrid_document_retriever_tool",
    # Aggiungi qui eventuali altri tool futuri
]

# Inizializziamo le metriche con le relative soglie calibrate
tool_hallucination = ToolHallucinationMetric(available_tools=TOOL_DISPONIBILI_MIMIR)
latenza = LatencyMetric(max_seconds=15.0)
token_usage = TokenEfficiencyMetric(max_tokens=25000)

lista_metriche = [agentic_synthesis, tool_hallucination, latenza, token_usage]


# ==========================================
# 4. CICLO DI TEST PYTEST
# ==========================================
@pytest.mark.parametrize("item", json.load(open("tracce_agente.json", "r", encoding="utf-8")))
def test_mimir_agent(item):
    # Prepariamo la risposta del tool. DeepEval vuole il retrieval_context come lista di stringhe.
    risposta_tool = item.get("risposta_del_tool", "")
    contesto_lista = [str(risposta_tool)] if risposta_tool else []

    # Costruiamo il test case mappando i campi estratti da Langfuse
    test_case = LLMTestCase(
        input=item.get("input", ""),
        actual_output=str(item.get("actual_output", "")),
        retrieval_context=contesto_lista,
        additional_metadata={
            "tool_called": item.get("tool_chiamato_effettivamente", ""),
            "latency_seconds": item.get("latency_seconds", 0),
            "total_tokens": item.get("total_tokens", 0)
        }
    )

    # --- ESECUZIONE PARALLELA ---
    start_time = time.perf_counter()

    async def run_metrics():
        # Eseguiamo tutte le metriche (sia custom che GEval) in contemporanea
        await asyncio.gather(*[m.a_measure(test_case) for m in lista_metriche])

    asyncio.run(run_metrics())

    end_time = time.perf_counter()
    duration = round(end_time - start_time, 2)

    # --- RACCOLTA E SALVATAGGIO RISULTATI ---
    results_to_save = []
    for metric in lista_metriche:
        real_status = "PASSED" if metric.score >= metric.threshold else "FAILED"
        nome_metrica = getattr(metric, "__name__", metric.__class__.__name__)

        results_to_save.append({
            "Input": str(item.get("input", ""))[:100],  # Tronca input lunghi per non intasare il CSV
            "Metrica": nome_metrica,
            "Punteggio": metric.score,
            "Soglia": metric.threshold,
            "Esito": real_status,
            "Durata (s)": duration,
            "Motivazione": str(metric.reason).replace("\n", " ").replace(";", ",")
        })

    df = pd.DataFrame(results_to_save)

    # Sistema di retry sicuro per il salvataggio concorrente (utile con pytest-xdist)
    for attempt in range(20):
        try:
            header = not os.path.exists(CSV_FILE)
            df.to_csv(CSV_FILE, mode='a', index=False, sep=";", encoding="utf-8-sig", header=header)
            break
        except Exception:
            time.sleep(0.5 + (attempt * 0.1))