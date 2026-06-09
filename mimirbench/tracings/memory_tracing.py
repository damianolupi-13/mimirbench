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

class MemoryTraceExtractor(BaseTraceExtractor):
    """
        Classe per scaricare e usare le traces Langfuse riferite alla valutazione mnemonica dell'agente
    """

    def __init__(self, langfuse_instance):
        super().__init__(langfuse_instance)

    def fetching(self, trace_output):
        """
            Parser per estrarre il puro dialogo verbale dall'output di una traccia
        """
        # trace_output deve essere un dizionario
        if not isinstance(trace_output, dict):
            return None, None

        # Estrazione del ruolo e contenuto dell'output della traccia
        tipo = trace_output.get("type") or trace_output.get("role")
        content = trace_output.get("content", "")

        # Gestione messaggio dell'utente
        if tipo in ["user", "human"]:
            return "user", content

        # Gestione messaggio dell'agente
        if tipo in ["assistant", "ai"]:
            if trace_output.get("tool_calls") or trace_output.get("additional_kwargs", {}).get("tool_calls"):
                return None, None #ignora le chiamate ai tool
            return "assistant", content

        # Gestione dei messaggi nel caso in cui il framework usato li gestisce come serializzazioni delle classi python con id
        if "kwargs" in trace_output and "id" in trace_output:
            class_name = trace_output["id"][-1] if isinstance(trace_output["id"], list) else str(trace_output["id"])
            content = trace_output["kwargs"].get("content", "")
            if "Human" in class_name or "User" in class_name: return "user", content
            if "AI" in class_name or "Assistant" in class_name:
                if trace_output["kwargs"].get("tool_calls") or trace_output["kwargs"].get("additional_kwargs", {}).get("tool_calls"):
                    return None, None
                return "assistant", content

        return None, None

    def extracting(self, output_json_path: str, test_id: str):
        """
            Estrazione della conversazione più recente e più lunga possibile, trovando la traccia che la contiene.
        """
        print("ESTRAZIONE CONVERSAZIONE COMPLETA da Langfuse...\n")

        try:
            import time
            res = None
            # Recuperiamo tutte le tracce con il tag
            # --- CONTROLLO RICERCA INIZIALE ---
            for tentativo in range(3):
                try:
                    # limit = 50, ci interessa trovare UNA traccia sola
                    res = self.langfuse_instance.api.trace.list(limit=50, tags=["env:test"])
                    break
                except Exception as err:
                    print(f"    [!] Timeout ricerca lista (Tentativo {tentativo + 1}/3). Ritento...")
                    time.sleep(3)

            if not res:
                print("ERRORE CRITICO: Impossibile contattare Langfuse per la lista iniziale.")
                return None

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

            print(f"Trovate {len(tracce_test)} tracce totali. Cerco la conversazione più lunga e recente...\n")

            # Invece di ordinare dalla più vecchia alla più nuova,
            # usiamo reverse=True per avere la PIÙ RECENTE in posizione 0
            tracce_ordinate = sorted(tracce_test, key=lambda x: getattr(x, 'timestamp', 0), reverse=True)

            conversazione_migliore = None

            # Analizziamo partendo dall'ultima conversazione fatta
            for t_info in tracce_ordinate:
                import time
                trace = None
                osservazioni = []

                print(f" -> Verifico la traccia {t_info.id}...")

                # --- CONTROLLO SINGOLA TRACCIA ---
                for tentativo in range(3):
                    try:
                        trace = self.langfuse_instance.api.trace.get(t_info.id)
                        osservazioni = sorted(trace.observations, key=lambda x: getattr(x, 'start_time', 0))
                        break
                    except Exception as err:
                        print(f"    [!] Timeout (Tentativo {tentativo + 1}/3). Ritento... {err}")
                        time.sleep(3)

                if not trace or not osservazioni:
                    print(f"    [!] Traccia {t_info.id} illeggibile, passo alla precedente.")
                    continue
                # -------------------------

                turni_della_traccia = []
                nodi_react_agent = [obs for obs in osservazioni if obs.name == "react_agent"]

                if nodi_react_agent:
                    ultimo_obs = nodi_react_agent[-1]
                    tutti_i_messaggi = []

                    if ultimo_obs.input and isinstance(ultimo_obs.input, dict) and "messages" in ultimo_obs.input:
                        tutti_i_messaggi.extend(ultimo_obs.input["messages"])
                    if ultimo_obs.output and isinstance(ultimo_obs.output, dict) and "messages" in ultimo_obs.output:
                        for m in ultimo_obs.output["messages"]:
                            if m not in tutti_i_messaggi:
                                tutti_i_messaggi.append(m)

                    domanda_tmp = ""
                    for msg in tutti_i_messaggi:
                        ruolo, testo = self.fetching(msg)
                        if ruolo == "user" and testo:
                            domanda_tmp = testo
                        elif ruolo == "assistant" and testo:
                            if domanda_tmp:
                                turni_della_traccia.append({
                                    "input": domanda_tmp,
                                    "actual_output": testo
                                })
                                domanda_tmp = ""

                numero_turni_attuali = len(turni_della_traccia)

                # Se la traccia recente che abbiamo appena scaricato ha almeno 2 turni,
                # LA SALVIAMO E INTERROMPIAMO IMMEDIATAMENTE IL CICLO
                if numero_turni_attuali >= 2:
                    conversazione_migliore = {
                        "id_traccia_finale": t_info.id,
                        "numero_turni": numero_turni_attuali,
                        "turni": turni_della_traccia
                    }
                    print(f"    [OK] Trovato storico completo ({numero_turni_attuali} turni). Interrompo la ricerca.")
                    break  # <--- BLOCCA IL CICLO FOR
                else:
                    # Se per caso l'ultima traccia era difettosa o vuota, il ciclo continua
                    # e andrà a scaricare la penultima.
                    print(f"    [-] La traccia aveva solo {numero_turni_attuali} turni. Controllo la precedente...")

            # --- ASSEMBLAGGIO FINALE ---
            if conversazione_migliore:
                print(
                    f"La traccia più completa e recente è la {conversazione_migliore['id_traccia_finale']} con {conversazione_migliore['numero_turni']} turni.")

                dati_da_salvare = [conversazione_migliore]

                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(dati_da_salvare, f, indent=4, ensure_ascii=False)

                print(f"Estrazione completata. JSON pronto.")
            else:
                print(
                    f"Nessuna traccia contiene una conversazione di almeno 2 turni. Impossibile valutare la memoria.\n")

            return None

        except Exception as e:
            print(f"ERRORE DURANTE L'ESTRAZIONE: {e}")