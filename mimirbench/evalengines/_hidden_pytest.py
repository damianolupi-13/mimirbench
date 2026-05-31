# Copyright 2026 Damiano Lupi (https://github.com/damianolupi-13)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ---
# MODIFICA: Questo file contiene codice originariamente derivato da
# confident-ai/deepeval (https://github.com/confident-ai/deepeval)
# ed è stato modificato o esteso per le specifiche di MimirBench.

import os
import json
import asyncio
import time
import pytest
import pandas as pd
import importlib
from dotenv import load_dotenv

from deepeval.test_case import LLMTestCase, ConversationalTestCase, Turn
from deepeval.metrics import (
    BaseMetric, FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric,
    KnowledgeRetentionMetric
)

JSON_PATH = os.environ.get("MIMIR_JSON_PATH")
CSV_FILE = os.environ.get("MIMIR_CSV_PATH")
EVAL_MODE = os.environ.get("MIMIR_EVAL_MODE")
METRICS_PROVIDER = os.environ.get("MIMIR_METRICS_PROVIDER", "DEFAULT")
TESTCASE_BUILDER = os.environ.get("MIMIR_TESTCASE_BUILDER", "DEFAULT")

load_dotenv("../../testdata/api_key.env")

# ==========================================
# METRICHE CUSTOM INTERNE (MIMIRBENCH CORE)
# ==========================================

class ToolHallucinationMetric(BaseMetric):
    def __init__(self, available_tools: list):
        self.available_tools = available_tools
        self.threshold = 1.0

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
        return getattr(self, 'success', False)

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
        return getattr(self, 'success', False)

    @property
    def __name__(self):
        return "Response Latency"


class TokenEfficiencyMetric(BaseMetric):
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.threshold = 1.0

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        actual_tokens = test_case.additional_metadata.get("total_tokens", 0)
        self.success = actual_tokens <= self.max_tokens
        self.score = 1.0 if self.success else 0.0
        self.reason = f"Token usati: {actual_tokens} (Soglia: {self.max_tokens})"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return getattr(self, 'success', False)

    @property
    def __name__(self):
        return "Token Efficiency"

# ==========================================
# CONFIGURAZIONE E RUNNER
# ==========================================

def get_metrics():
    """Carica le metriche dinamicamente con logica strict fail-fast."""
    if METRICS_PROVIDER != "DEFAULT":
        modulo_nome, funzione_nome = METRICS_PROVIDER.split(":")
        modulo = importlib.import_module(modulo_nome)
        return getattr(modulo, funzione_nome)()

    if EVAL_MODE == "RAG":
        return [
            FaithfulnessMetric(threshold=0.7, model="gpt-4o-mini", async_mode=True),
            AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini", async_mode=True),
            ContextualRelevancyMetric(threshold=0.4, model="gpt-4o-mini", async_mode=True)
        ]
    elif EVAL_MODE == "MEMORY":
        return [KnowledgeRetentionMetric(threshold=0.7, model="gpt-4o-mini", async_mode=True)]
    elif EVAL_MODE == "AGENT":
        return [
            ToolHallucinationMetric(available_tools=["hybrid_document_retriever_tool", "web_search", "calculator"]),
            LatencyMetric(max_seconds=15.0),
            TokenEfficiencyMetric(max_tokens=25000)
        ]

    raise ValueError(
        f"Nessuna metrica di default per la modalità '{EVAL_MODE}'. "
        "Devi fornire un 'metrics_provider' all'Engine."
    )

def configure_test_case(item):
    """Costruisce il Test Case, supportando mappature custom."""
    if TESTCASE_BUILDER != "DEFAULT":
        modulo_nome, funzione_nome = TESTCASE_BUILDER.split(":")
        modulo = importlib.import_module(modulo_nome)
        costruttore_custom = getattr(modulo, funzione_nome)
        return costruttore_custom(item)

    if EVAL_MODE == "RAG":
        return LLMTestCase(
            input=item.get("input", ""), actual_output=item.get("actual_output", ""), retrieval_context=item.get("retrieval_context", [])
        )

    elif EVAL_MODE == "AGENT":
        contesto = [str(item.get("risposta_del_tool", ""))] if item.get("risposta_del_tool") else []
        return LLMTestCase(
            input=item.get("input", ""), actual_output=str(item.get("actual_output", "")), retrieval_context=contesto,
            additional_metadata={
                "tool_called": item.get("tool_chiamato_effettivamente", ""),
                "latency_seconds": item.get("latency_seconds", 0),
                "total_tokens": item.get("total_tokens", 0)
            }
        )

    elif EVAL_MODE == "MEMORY":
        turns = [Turn(role="user", content=t["input"]) for t in item.get("turni", [])] + \
                [Turn(role="assistant", content=t["actual_output"]) for t in item.get("turni", [])]
        return ConversationalTestCase(turns=turns)

    # Fallback generico per custom EVAL_MODE senza un builder specifico
    return LLMTestCase(input=item.get("input", ""), actual_output=str(item.get("actual_output", "")))

METRICHE_DA_ESEGUIRE = get_metrics()

@pytest.mark.parametrize("item", json.load(open(JSON_PATH, "r", encoding="utf-8")))
def test_mimir_universal(item):
    if not METRICHE_DA_ESEGUIRE:
        pytest.skip("Nessuna metrica fornita per questo test. Esecuzione saltata.")

    test_case = configure_test_case(item)
    start_time = time.perf_counter()

    async def run_metrics():
        await asyncio.gather(*[m.a_measure(test_case) for m in METRICHE_DA_ESEGUIRE])

    asyncio.run(run_metrics())
    duration = round(time.perf_counter() - start_time, 2)

    results_to_save = []
    input_ref = str(item.get("id_traccia_finale", str(item.get("input", ""))[:100]))

    for metric in METRICHE_DA_ESEGUIRE:
        results_to_save.append({
            "Riferimento": input_ref,
            "Metrica": getattr(metric, "__name__", metric.__class__.__name__),
            "Punteggio": metric.score,
            "Soglia": getattr(metric, 'threshold', 'N/A'),
            "Esito": "PASSED" if metric.score >= metric.threshold else "FAILED",
            "Durata (s)": duration,
            "Motivazione": str(metric.reason).replace("\n", " ").replace(";", ",")
        })

    df = pd.DataFrame(results_to_save)
    for attempt in range(20):
        try:
            header = not os.path.exists(CSV_FILE)
            df.to_csv(CSV_FILE, mode='a', index=False, sep=";", encoding="utf-8-sig", header=header)
            break
        except Exception:
            time.sleep(0.5 + (attempt * 0.1))