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

import os
import subprocess
from abc import ABC


class BaseEvalEngine(ABC):
    """
    Motore base per le valutazioni. Prepara l'ambiente isolato e lancia lo script Pytest.
    Supporta l'estensione tramite provider di metriche e builder di Test Case personalizzati.
    """

    def __init__(self, json_dati_path: str, output_csv_path: str,
                 metrics_provider: str = None,
                 testcase_builder: str = None,
                 custom_test_script: str = None):

        self.json_dati = json_dati_path
        self.output_csv = output_csv_path
        self.metrics_provider = metrics_provider or "DEFAULT"
        self.testcase_builder = testcase_builder or "DEFAULT"

        # Gestione della Botola (Escape Hatch)
        if custom_test_script and os.path.exists(custom_test_script):
            self.test_script = custom_test_script
        else:
            cartella_corrente = os.path.dirname(os.path.abspath(__file__))
            self.test_script = os.path.join(cartella_corrente, "test_hidden_pytest.py")

    def run(self, eval_mode: str = None):
        # Clona l'ambiente per non inquinare il sistema globale
        custom_env = os.environ.copy()

        # Disabilita telemetria e login forzato di DeepEval
        custom_env["CONFIDENT_METRIC_LOGGING_VERBOSE"] = "0"
        custom_env["DEEPEVAL_DISABLE_METRIC_LOGGING"] = "YES"
        custom_env["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"
        custom_env["CONFIDENT_API_KEY"] = "None"
        custom_env["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "300"

        # Passa le configurazioni al subprocess
        custom_env["MIMIR_JSON_PATH"] = self.json_dati
        custom_env["MIMIR_CSV_PATH"] = self.output_csv
        custom_env["MIMIR_EVAL_MODE"] = eval_mode
        custom_env["MIMIR_METRICS_PROVIDER"] = self.metrics_provider
        custom_env["MIMIR_TESTCASE_BUILDER"] = self.testcase_builder

        if os.path.exists(self.output_csv):
            try:
                os.remove(self.output_csv)
            except Exception:
                pass

        # Lancio del processo isolato
        subprocess.run(["deepeval", "test", "run", self.test_script], text=True, env=custom_env)


# --- CLASSI PUBBLICHE (Batteries Included) ---

class RagEvalEngine(BaseEvalEngine):
    def run_evaluations(self):
        self.run(eval_mode="RAG")


class AgentEvalEngine(BaseEvalEngine):
    def run_evaluations(self):
        self.run(eval_mode="AGENT")


class MemoryEvalEngine(BaseEvalEngine):
    def run_evaluations(self):
        self.run(eval_mode="MEMORY")