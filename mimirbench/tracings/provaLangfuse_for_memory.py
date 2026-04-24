import os
import json
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv("../../langfusekeys.env")
langfuse = Langfuse()

# --- INSERISCI QUI IL TAG CORRETTO ---
TAG_DA_CERCARE = "prova-Langfuse-4-tesi-15mar"


def estrai_ruolo_e_testo(msg):
    if not isinstance(msg, dict):
        return None, None

    tipo = msg.get("type") or msg.get("role")
    content = msg.get("content", "")

    if tipo in ["user", "human"]:
        return "user", content

    if tipo in ["assistant", "ai"]:
        if msg.get("tool_calls") or msg.get("additional_kwargs", {}).get("tool_calls"):
            return None, None
        return "assistant", content

    if "kwargs" in msg and "id" in msg:
        class_name = msg["id"][-1] if isinstance(msg["id"], list) else str(msg["id"])
        content = msg["kwargs"].get("content", "")
        if "Human" in class_name or "User" in class_name: return "user", content
        if "AI" in class_name or "Assistant" in class_name:
            if msg["kwargs"].get("tool_calls") or msg["kwargs"].get("additional_kwargs", {}).get("tool_calls"):
                return None, None
            return "assistant", content

    return None, None


print("🚀 ESTRAZIONE CONVERSAZIONE COMPLETA da Langfuse...\n")

try:
    # 1. Recuperiamo tutte le tracce con il tag
    res = langfuse.api.trace.list(limit=100, tags=[TAG_DA_CERCARE])

    if not res.data:
        print("❌ Nessuna traccia trovata con questo tag.")
        exit()

    print(f"🔍 Trovate {len(res.data)} tracce totali. Cerco la conversazione più lunga e recente...\n")

    conversazione_migliore = None
    max_turni_trovati = 0

    # 2. Ordiniamo le tracce dalla più VECCHIA alla più NUOVA
    # Così se usiamo il ">=", la più recente sovrascriverà quella vecchia a parità di turni
    tracce_ordinate = sorted(res.data, key=lambda x: getattr(x, 'timestamp', 0))

    # 3. Analizziamo TUTTE le tracce in ordine cronologico
    for t_info in tracce_ordinate:
        trace = langfuse.api.trace.get(t_info.id)
        turni_della_traccia = []

        osservazioni = sorted(trace.observations, key=lambda x: getattr(x, 'start_time', 0))
        nodi_react_agent = [obs for obs in osservazioni if obs.name == "react_agent"]

        if nodi_react_agent:
            # Prendiamo solo l'ultimo passaggio nel react_agent per evitare loop dei tools
            ultimo_obs = nodi_react_agent[-1]
            tutti_i_messaggi = []

            if ultimo_obs.input and isinstance(ultimo_obs.input, dict) and "messages" in ultimo_obs.input:
                tutti_i_messaggi.extend(ultimo_obs.input["messages"])

            if ultimo_obs.output and isinstance(ultimo_obs.output, dict) and "messages" in ultimo_obs.output:
                for m in ultimo_obs.output["messages"]:
                    if m not in tutti_i_messaggi:
                        tutti_i_messaggi.append(m)

            # Ricostruiamo le coppie domanda/risposta
            domanda_tmp = ""
            for msg in tutti_i_messaggi:
                ruolo, testo = estrai_ruolo_e_testo(msg)

                if ruolo == "user" and testo:
                    domanda_tmp = testo
                elif ruolo == "assistant" and testo:
                    if domanda_tmp:
                        turni_della_traccia.append({
                            "input": domanda_tmp,
                            "actual_output": testo
                        })
                        domanda_tmp = ""

                        # 4. IL TUO FIX: usiamo ">=" per prendere la più lunga e, a parità, la più recente!
        numero_turni_attuali = len(turni_della_traccia)
        if numero_turni_attuali >= max_turni_trovati and numero_turni_attuali >= 2:
            max_turni_trovati = numero_turni_attuali
            conversazione_migliore = {
                "id_traccia_finale": t_info.id,
                "numero_turni": numero_turni_attuali,
                "turni": turni_della_traccia
            }

    # --- ASSEMBLAGGIO FINALE ---
    if conversazione_migliore:
        print(
            f"✅ TROVATA! La traccia più completa (e recente) è la {conversazione_migliore['id_traccia_finale']} con {conversazione_migliore['numero_turni']} turni.")

        dati_da_salvare = [conversazione_migliore]

        with open("conversazioni_memoria.json", "w", encoding="utf-8") as f:
            json.dump(dati_da_salvare, f, indent=4, ensure_ascii=False)

        print(f"💾 Estrazione completata! JSON pronto per DeepEval.")
    else:
        print(f"⚠️ Nessuna traccia contiene una conversazione di almeno 2 turni. Impossibile valutare la memoria.\n")

except Exception as e:
    print(f"❌ ERRORE DURANTE L'ESTRAZIONE: {e}")