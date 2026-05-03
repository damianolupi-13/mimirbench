#Implementare classe ContextualTraceExtractor

import json
from mimirbench.tracings.base_tracing import BaseTraceExtractor

class ContextualTraceExtractor(BaseTraceExtractor):
    """
        Classe per scaricare e usare le traces Langfuse riferite alla valutazione contestuale dell'agente
    """

    def __init__(self, tracing_tag: str):
        super().__init__(tracing_tag)

    def fetching(self, trace_output):
        """Estrae il testo se è una risposta finale (messaggio AI senza tool calls)"""
        if isinstance(trace_output, dict) and "messages" in trace_output:
            last_msg = trace_output["messages"][-1]
            # Se non c'è una chiamata a tool, questa è la nostra risposta
            if not last_msg.get("additional_kwargs", {}).get("function_call") and not last_msg.get("tool_calls"):
                return last_msg.get("content")
        return None

    def extracting(self, output_json_path: str):
        print("ESTRAZIONE DIRETTA da Langfuse (Numerazione in ordine cronologico inverso)...\n")
        dati_estratti = []  # Lista che conterrà i dizionari da salvare in JSON

        try:
            res = self.langfuse_instance.api.trace.list(limit=100, tags=[self.tracing_tag])

            for i, t_info in enumerate(res.data):
                trace = self.langfuse_instance.api.trace.get(t_info.id)

                faldone_documenti = []
                risposta_finale = ""
                domanda_utente = ""

                # CICLO DIRETTO: Langfuse restituisce i dati dal più recente al più vecchio
                for obs in trace.observations:
                    if obs.name == "react_agent":
                        testo = self.fetching(obs.output)
                        if testo:  # Manteniamo un controllo su quale effettivamente delle fasi react_agent formula la risposta
                            # Abbiamo trovato l'ultimo react_agent (quello della risposta)
                            input_data = getattr(obs, 'input', {})
                            if isinstance(input_data, dict):
                                faldone_documenti = input_data.get("documents", [])
                                risposta_finale = testo

                                # Domanda dell'utente (ultimo messaggio human nello stato)
                                msgs = input_data.get("messages", [])
                                for m in reversed(msgs):
                                    if m.get("type") == "human":
                                        domanda_utente = m.get("content")
                                        break
                            break  # Trovato l'ultimo, possiamo passare alla prossima traccia

                # --- OUTPUT ---
                print(f"==================== TRACCIA {i + 1} ====================")
                print(f"DOMANDA: {domanda_utente}")
                print(f"RISPOSTA: {risposta_finale[:150]}...")
                print(f"DOCUMENTI: {len(faldone_documenti)}")

                if faldone_documenti:
                    for idx, doc in enumerate(faldone_documenti):
                        # Estraiamo l'ID e la pagina dai metadati
                        meta = doc.get("metadata", {})
                        pag = meta.get("page_numbers", ["?"])[0]
                        print(f"   [{idx + 1}] Pag. {pag} | {doc.get('page_content', '')[:80]}...")
                print("=" * 60 + "\n")

                # Estrazione informazioni per DeepEval
                if domanda_utente and risposta_finale and faldone_documenti:
                    contesti_testuali = [doc.get("page_content", "") for doc in faldone_documenti]

                    # Creiamo un dizionario per questa traccia
                    traccia_data = {
                        "id_traccia": t_info.id,
                        "input": domanda_utente,
                        "actual_output": risposta_finale,
                        "retrieval_context": contesti_testuali
                    }
                    dati_estratti.append(traccia_data)
                    print(f"Traccia {i + 1} elaborata.")

            # Salvataggio su file JSON
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(dati_estratti, f, indent=4, ensure_ascii=False)

            print(f"\nEstrazione completata! {len(dati_estratti)} tracce salvate in: {output_json_path}\n")

        except Exception as e:
            print(f"ERRORE: {e}")


