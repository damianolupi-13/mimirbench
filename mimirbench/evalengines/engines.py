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
import json
import subprocess
from abc import ABC, abstractmethod


class BaseEvalEngine(ABC):
    """
    Motore base astratto per le valutazioni di Mimir.
    Fornisce la configurazione di base dell'ambiente, ma delega
    l'esecuzione effettiva alle implementazioni specifiche.
    """

    def __init__(self, json_dati_path: str = "", output_csv_path: str = "", parallel_launches: int = 1, provider: str = "openai",
                 model: str = "gpt-5-nano"):
            # Convertiamo i path in percorsi ASSOLUTI basati su dove viene lanciato lo script principale
            self._json_dati_path = os.path.abspath(json_dati_path) if json_dati_path else ""
            self._output_csv_path = os.path.abspath(output_csv_path) if output_csv_path else ""
            self._parallel_launches = parallel_launches
            self._provider = provider.lower()
            self._model = model

    def _get_base_env(self) -> dict:
        """
        Fornisce il dizionario con le variabili d'ambiente clonate e
        configurate per disabilitare telemetria e login di DeepEval.
        """
        custom_env = os.environ.copy()

        # --- GESTIONE GENERICA AMBIENTE ---
        print("\n=== CONFIGURAZIONE AMBIENTE MIMIR ===")
        print(f"Provider Giudice Selezionato: {self._provider.upper()}")
        print(f"API_KEY Globale impostata: {'API_KEY' in custom_env}")
        print("=====================================")

        custom_env = os.environ.copy()
        custom_env["CONFIDENT_METRIC_LOGGING_VERBOSE"] = "0"
        custom_env["DEEPEVAL_DISABLE_METRIC_LOGGING"] = "YES"
        custom_env["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"
        custom_env["CONFIDENT_API_KEY"] = "None"
        custom_env["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "300"
        custom_env["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = "600"

        return custom_env

    @abstractmethod
    def run(self):
        """
        Metodo astratto. Ogni Engine DEVE implementare questo metodo
        per definire come lanciare il proprio specifico file di test.
        """
        pass


# --- Implementazioni degli Engine ---

class RagEvalEngine(BaseEvalEngine):
    """Motore dedicato alle metriche RAG tradizionali"""

    def __init__(self, json_dati_path: str = "", output_csv_path: str = "", parallel_launches: int = 5, provider: str = "openai",
                 model: str = "gpt-5-nano"):
        super().__init__(json_dati_path, output_csv_path, parallel_launches, provider, model)
        self._test_path = "test_RAGMetrics.py"

    def run(self):
        cartella_corrente = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(cartella_corrente, self._test_path)
        config_path = os.path.join(cartella_corrente, "mimir_eval_config.json")

        if not os.path.exists(script_path):
            raise FileNotFoundError(f"\n[ERRORE MIMIR ENGINE] Script di test non trovato: {script_path}")

        print(f"\n[MIMIR ENGINE] Avvio RagEvalEngine sul file: {self._test_path}")

        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY", "")

        config_data = {
            "MIMIR_JSON_PATH": self._json_dati_path,
            "MIMIR_CSV_PATH": self._output_csv_path,
            "MIMIR_EVAL_PROVIDER": self._provider,
            "MIMIR_EVAL_MODEL": self._model,
            "MIMIR_EVAL_API_KEY": api_key
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)

        comando = ["deepeval", "test", "run", script_path]
        if self._parallel_launches > 1:
            comando.extend(["-n", str(self._parallel_launches)])

        print(f"\n[MIMIR ENGINE] Esecuzione PARALLELA attivata: {self._parallel_launches} worker")

        # Il comando più normale e pulito possibile
        try:
            subprocess.run(comando, text=True, env=self._get_base_env(), cwd=cartella_corrente)
        finally:
            if os.path.exists(config_path):
                os.remove(config_path)

class AgentEvalEngine(BaseEvalEngine):
    """Motore dedicato alle performance tecniche e di ragionamento degli agenti AI"""
    def __init__(self, json_dati_path: str = "", output_csv_path: str = "", parallel_launches: int = 1, tool_list: list = None,
                 provider: str = "openai", model: str = "gpt-5-nano"):
        super().__init__(json_dati_path, output_csv_path, parallel_launches, provider, model)
        self._test_path = "test_agent_metrics.py"
        self._tool_list = tool_list if tool_list is not None else []

    def run(self):
        cartella_corrente = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(cartella_corrente, self._test_path)
        config_path = os.path.join(cartella_corrente, "mimir_config.json")

        if not os.path.exists(script_path):
            raise FileNotFoundError(f"\n[ERRORE MIMIR ENGINE] Script di test non trovato: {script_path}")

        print(f"\n[MIMIR ENGINE] Avvio AgentEvalEngine sul file: {self._test_path}")

        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY", "")

        config_data = {
            "MIMIR_JSON_PATH": self._json_dati_path,
            "MIMIR_CSV_PATH": self._output_csv_path,
            "MIMIR_AVAILABLE_TOOLS": self._tool_list,
            "MIMIR_EVAL_PROVIDER": self._provider,
            "MIMIR_EVAL_MODEL": self._model,
            "MIMIR_EVAL_API_KEY": api_key
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)

        comando = ["deepeval", "test", "run", script_path]
        if self._parallel_launches > 1:
            comando.extend(["-n", str(self._parallel_launches)])

        print(f"\n[MIMIR ENGINE] Esecuzione PARALLELA attivata: {self._parallel_launches} worker")

        try:
            subprocess.run(comando, text=True, env=self._get_base_env(), cwd=cartella_corrente)
        finally:
            if os.path.exists(config_path):
                os.remove(config_path)


class MemoryEvalEngine(BaseEvalEngine):
    """Motore dedicato alla validazione della Knowledge Retention"""
    def __init__(self, json_dati_path: str = "", output_csv_path: str = "", parallel_launches: int = 1, provider: str = "openai",
                 model: str = "gpt-5-nano"):
        super().__init__(json_dati_path, output_csv_path, parallel_launches, provider, model)
        self._test_path = "test_memory_metric.py"

    def run(self):
        cartella_corrente = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(cartella_corrente, self._test_path)
        config_path = os.path.join(cartella_corrente, "mimir_config.json")

        if not os.path.exists(script_path):
            raise FileNotFoundError(f"\n[ERRORE MIMIR ENGINE] Script di test non trovato: {script_path}")

        print(f"\n[MIMIR ENGINE] Avvio MemoryEvalEngine sul file: {self._test_path}")

        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY", "")

        config_data = {
            "MIMIR_JSON_PATH": self._json_dati_path,
            "MIMIR_CSV_PATH": self._output_csv_path,
            "MIMIR_EVAL_PROVIDER": self._provider,
            "MIMIR_EVAL_MODEL": self._model,
            "MIMIR_EVAL_API_KEY": api_key
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)

        comando = ["deepeval", "test", "run", script_path]
        if self._parallel_launches > 1:
            comando.extend(["-n", str(self._parallel_launches)])

        print(f"\n[MIMIR ENGINE] Esecuzione PARALLELA attivata: {self._parallel_launches} worker")

        try:
            subprocess.run(comando, text=True, env=self._get_base_env(), cwd=cartella_corrente)
        finally:
            if os.path.exists(config_path):
                os.remove(config_path)