import os
import json
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv("../../langfusekeys.env")
langfuse = Langfuse()

# --- filtro tag
TAG_DA_CERCARE = "prova-Langfuse-2-tesi-02mar"


def get_final_answer(obs_output):
    """Estrae il testo se è una risposta finale (AI senza tool calls)"""
    if isinstance(obs_output, dict) and "messages" in obs_output:
        last_msg = obs_output["messages"][-1]
        # Se non c'è una chiamata a tool, questa è la nostra risposta
        if not last_msg.get("additional_kwargs", {}).get("function_call") and not last_msg.get("tool_calls"):
            return last_msg.get("content")
    return None


print("🚀 ESTRAZIONE DIRETTA da Langfuse (Numerazione in ordine cronologico inverso)...\n")
dati_estratti = [] # Lista che conterrà i dizionari da salvare in JSON

try:
    res = langfuse.api.trace.list(limit=100, tags=[TAG_DA_CERCARE])

    for i, t_info in enumerate(res.data):
        trace = langfuse.api.trace.get(t_info.id)

        faldone_reale = []
        risposta_finale = ""
        domanda_utente = ""

        # CICLO DIRETTO: Langfuse restituisce i dati dal più recente al più vecchio
        for obs in trace.observations:
            if obs.name == "react_agent":
                testo = get_final_answer(obs.output)
                if testo: #Manteniamo un controllo su quale effettivamente dei react_agent formula la risposta
                    # Abbiamo trovato l'ultimo react_agent (quello della risposta)
                    input_data = getattr(obs, 'input', {})
                    if isinstance(input_data, dict):
                        faldone_reale = input_data.get("documents", [])
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
        print(f"❓ DOMANDA: {domanda_utente}")
        print(f"💡 RISPOSTA: {risposta_finale[:150]}...")
        print(f"📄 DOCUMENTI: {len(faldone_reale)}")

        if faldone_reale:
            for idx, doc in enumerate(faldone_reale):
                # Estraiamo l'ID e la pagina dai metadati
                meta = doc.get("metadata", {})
                pag = meta.get("page_numbers", ["?"])[0]
                print(f"   [{idx + 1}] Pag. {pag} | {doc.get('page_content', '')[:80]}...")
        print("=" * 60 + "\n")

        #Estrazione informazioni per DeepEval
        if domanda_utente and risposta_finale and faldone_reale:
            contesti_testuali = [doc.get("page_content", "") for doc in faldone_reale]

            # Creiamo un dizionario per questa traccia
            traccia_data = {
                "id_traccia": t_info.id,
                "input": domanda_utente,
                "actual_output": risposta_finale,
                "retrieval_context": contesti_testuali
            }
            dati_estratti.append(traccia_data)
            print(f"✅ Traccia {i + 1} elaborata.")

    # Salvataggio su file JSON
    with open("dati_valutazione.json", "w", encoding="utf-8") as f:
        json.dump(dati_estratti, f, indent=4, ensure_ascii=False)

    print(f"\n💾 Estrazione completata! {len(dati_estratti)} tracce salvate in 'dati_valutazione.json'.")

except Exception as e:
    print(f"❌ ERRORE: {e}")