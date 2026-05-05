#Implementare TechnicalTraceExtractor

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

import json
from mimirbench.tracings.base_tracing import BaseTraceExtractor

class TechnicalTraceExtractor(BaseTraceExtractor):
    """
        Classe per scaricare e usare le traces Langfuse riferite alla valutazione tecnica dell'agente
    """
    def __init__(self):
        super().__init__()

    # Metodo per il fetching di determinate caratteristiche dell'output non serve
    def fetching(self,  trace_output):
        return None

    def extracting(self, output_json_path: str, test_id: str):
        print("🚀 ESTRAZIONE DATI AGENTE da Langfuse...\n")
        dati_estratti = []

        try:
            res = self.langfuse_instance.api.trace.list(limit=100, tags=["env:test"])

            if not res.data:
                print("Nessuna traccia trovata con il tag env:test.")
                return None

            tracce_test = [
                tr for tr in res.data
                if tr.metadata and tr.metadata.get("test_id") == test_id
            ]

            if not tracce_test:
                print(f"Nessuna traccia trovata con il test_id: {test_id}")
                return None

            print(f"Trovate {len(tracce_test)} tracce relative a questo test...\n")

            for i, t_info in enumerate(tracce_test):
                trace = self.langfuse_instance.api.trace.get(t_info.id)

                domanda_utente = ""
                risposta_finale = ""
                tool_chiamato = "Nessun tool"
                risposta_del_tool = ""

                # Metriche Tecniche
                latenza_sec = getattr(trace, "latency", 0)
                total_tokens = 0  # Inizializziamo a 0, lo calcoleremo noi

                # Ordinamento cronologico delle osservazioni della traccia analizzata dalla più vecchia alla più nuova
                osservazioni = sorted(trace.observations, key=lambda x: getattr(x, 'start_time', 0))

                for obs in osservazioni:
                    # --- CALCOLO TOKEN (Dal nodo ChatVertexAI) ---
                    # Identifichiamo il nodo che fa la chiamata LLM (come dal tuo screenshot)
                    if obs.name == "ChatVertexAI" or getattr(obs, "type", "") == "GENERATION":
                        uso = getattr(obs, "usage", None)
                        if uso:
                            # Metodo corazzato: controlliamo sia come oggetto che come dizionario
                            if isinstance(uso, dict):
                                # Cerca tutte le varianti note
                                total_tokens += uso.get("total", 0) or uso.get("total_tokens", 0) or uso.get(
                                    "totalTokens", 0)
                            else:
                                # Se è un oggetto (standard SDK Langfuse)
                                if hasattr(uso, "total") and uso.total:
                                    total_tokens += uso.total
                                elif hasattr(uso, "total_tokens") and uso.total_tokens:
                                    total_tokens += uso.total_tokens

                    # --- CERCHIAMO LA DOMANDA UTENTE ---
                    if not domanda_utente and obs.input and isinstance(obs.input, dict):
                        messages = obs.input.get("messages", [])
                        if isinstance(messages, list):
                            for m in messages:
                                if isinstance(m, dict) and m.get("type") == "human":
                                    domanda_utente = m.get("content")
                                    break

                    # --- CERCHIAMO IL NODO "tools" ---
                    if obs.name == "tools" and obs.output and isinstance(obs.output, dict):
                        messages = obs.output.get("messages", [])
                        for msg in messages:
                            if isinstance(msg, dict) and msg.get("type") == "tool":
                                tool_chiamato = msg.get("name", "Tool_Sconosciuto")
                                content = msg.get("content", "")
                                risposta_del_tool = content if isinstance(content, str) else json.dumps(content,
                                                                                                        ensure_ascii=False)

                    # --- CERCHIAMO IL NODO "react_agent" (Risposta discorsiva finale) ---
                    if obs.name == "react_agent" and obs.output and isinstance(obs.output, dict):
                        messages = obs.output.get("messages", [])
                        for msg in messages:
                            if isinstance(msg, dict) and msg.get("type") == "ai":
                                # E NON contiene richieste di chiamare altri tool
                                if not msg.get("tool_calls") and not msg.get("additional_kwargs", {}).get("tool_calls"):
                                    risposta_finale = msg.get("content")

                # --- STAMPA A SCHERMO PER DEBUG ---
                print(f"==================== TRACCIA {i + 1} ====================")
                print(f"DOMANDA: {domanda_utente}")
                print(f"TOOL SCELTO: {tool_chiamato}")
                if risposta_del_tool:
                    print(f"RISPOSTA TOOL: {str(risposta_del_tool)[:80]}...")
                print(f"RISPOSTA FINALE: {str(risposta_finale)[:150]}...")
                print(f"LATENZA: {latenza_sec}s | TOKEN: {total_tokens}")
                print("=" * 60 + "\n")

                # --- ASSEMBLAGGIO JSON PER DEEPEVAL ---
                if domanda_utente and risposta_finale:
                    traccia_data = {
                        "id_traccia": t_info.id,
                        "input": domanda_utente,
                        "actual_output": risposta_finale,
                        "tool_chiamato_effettivamente": tool_chiamato,
                        "risposta_del_tool": risposta_del_tool,
                        "latency_seconds": latenza_sec,
                        "total_tokens": total_tokens
                    }
                    dati_estratti.append(traccia_data)

            # --- SALVATAGGIO SU FILE ---
            with open("tracce_agente.json", "w", encoding="utf-8") as f:
                json.dump(dati_estratti, f, indent=4, ensure_ascii=False)

            print(f"\nEstrazione completata {len(dati_estratti)} tracce salvate in 'tracce_agente.json'.")

        except Exception as e:
            print(f"ERRORE DURANTE L'ESTRAZIONE: {e}")