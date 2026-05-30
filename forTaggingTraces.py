#SCRIPT PER TAGGARE LE TRACES DOPO GLI ESPERIMENTI DEL GIORNO
import os
import requests
import uuid
from dotenv import load_dotenv
from langfuse import Langfuse
from datetime import datetime

# 1. Caricamento chiavi
load_dotenv("testdata/langfusekeys.env")
PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

langfuse = Langfuse()

# IMPOSTA IL GIORNO E IL TAG
GIORNO_TARGET = "2026-03-15"
NUOVO_TAG = "prova-Langfuse-4-tesi-15mar"

target_date = datetime.strptime(GIORNO_TARGET, "%Y-%m-%d").date()

print(f"🏷️  Inizio tagging via Ingestion API (POST) per il giorno: {GIORNO_TARGET}...")

try:
    # 2. Recuperiamo le tracce (per leggere l'SDK funziona bene)
    res = langfuse.api.trace.list(limit=50)

    count = 0
    for trace_data in res.data:
        if trace_data.timestamp.date() == target_date: #and trace_data.name == "LangGraph":

            tag_attuali = trace_data.tags if trace_data.tags else []

            if NUOVO_TAG not in tag_attuali:
                tag_attuali.append(NUOVO_TAG)

                print(f"✅ Invio update per traccia {trace_data.id[:8]}...")

                # --- LOGICA DI INGESTION (Merge via POST) ---
                url = f"{HOST}/api/public/ingestion"

                # Struttura richiesta da Langfuse per aggiornare/creare oggetti
                payload = {
                    "batch": [
                        {
                            "type": "trace-create",
                            "id": str(uuid.uuid4()),  # ID univoco per l'evento di update
                            "timestamp": datetime.now().isoformat() + "Z",
                            "body": {
                                "id": trace_data.id,  # L'ID della traccia esistente da colpire
                                "tags": tag_attuali,
                                "name": trace_data.name
                            }
                        }
                    ]
                }

                response = requests.post(
                    url,
                    auth=(PUBLIC_KEY, SECRET_KEY),
                    json=payload
                )

                if response.status_code in [200, 201, 207]:
                    count += 1
                else:
                    print(f"   ⚠️ Errore API: {response.status_code} - {response.text}")

    print(f"\n✨ Operazione completata! Inviati {count} aggiornamenti.")

except Exception as e:
    print(f"❌ ERRORE: {e}")