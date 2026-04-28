#Implementare TechnicalTraceExtractor

import json
from mimirbench.tracings.base_tracing import BaseTraceExtractor

class TechnicalTraceExtractor(BaseTraceExtractor):
    """
        Classe per scaricare e usare le traces Langfuse riferite alla valutazione tecnica dell'agente
    """
    def __init__(self, tracing_tag: str):
        super().__init__(tracing_tag)

    def fetching(self,  trace_output):
        return None

    def extracting(self, output_json_path: str):
        print("🚀 ESTRAZIONE DATI AGENTE da Langfuse...\n")
        dati_estratti = []

        try:
            res = self.langfuse_instance.api.trace.list(limit=100, tags=[self.tracing_tag])

            for i, t_info in enumerate(res.data):
                trace = self.langfuse_instance.api.trace.get(t_info.id)

                domanda_utente = ""
                risposta_finale = ""
                tool_chiamato = "Nessun tool"
                risposta_del_tool = ""

                # Metriche Tecniche
                latenza_sec = getattr(trace, "latency", 0)
                total_tokens = 0  # Inizializziamo a 0, lo calcoleremo noi

                # Ordinamento cronologico
                osservazioni = sorted(trace.observations, key=lambda x: getattr(x, 'start_time', 0))

                for obs in osservazioni:

                    # --- 0. CALCOLO TOKEN (Dal nodo ChatVertexAI) ---
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

                    # --- 1. CERCHIAMO LA DOMANDA UTENTE ---
                    if not domanda_utente and obs.input and isinstance(obs.input, dict):
                        messages = obs.input.get("messages", [])
                        if isinstance(messages, list):
                            for m in messages:
                                if isinstance(m, dict) and m.get("type") == "human":
                                    domanda_utente = m.get("content")
                                    break

                    # --- 2. CERCHIAMO IL NODO "tools" ---
                    if obs.name == "tools" and obs.output and isinstance(obs.output, dict):
                        messages = obs.output.get("messages", [])
                        for msg in messages:
                            if isinstance(msg, dict) and msg.get("type") == "tool":
                                tool_chiamato = msg.get("name", "Tool_Sconosciuto")
                                content = msg.get("content", "")
                                risposta_del_tool = content if isinstance(content, str) else json.dumps(content,
                                                                                                        ensure_ascii=False)

                    # --- 3. CERCHIAMO IL NODO "react_agent" (Risposta discorsiva finale) ---
                    if obs.name == "react_agent" and obs.output and isinstance(obs.output, dict):
                        messages = obs.output.get("messages", [])
                        for msg in messages:
                            if isinstance(msg, dict) and msg.get("type") == "ai":
                                # E NON contiene richieste di chiamare altri tool
                                if not msg.get("tool_calls") and not msg.get("additional_kwargs", {}).get("tool_calls"):
                                    risposta_finale = msg.get("content")

                # --- STAMPA A SCHERMO PER DEBUG ---
                print(f"==================== TRACCIA {i + 1} ====================")
                print(f"❓ DOMANDA: {domanda_utente}")
                print(f"🛠️  TOOL SCELTO: {tool_chiamato}")
                if risposta_del_tool:
                    print(f"📥 RISPOSTA TOOL: {str(risposta_del_tool)[:80]}...")
                print(f"💡 RISPOSTA FINALE: {str(risposta_finale)[:150]}...")
                print(f"⏱️  LATENZA: {latenza_sec}s | 🪙 TOKEN: {total_tokens}")
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

            print(f"\n💾 Estrazione completata! {len(dati_estratti)} tracce salvate in 'tracce_agente.json'.")

        except Exception as e:
            print(f"❌ ERRORE DURANTE L'ESTRAZIONE: {e}")