import sys
import os

# Forza Python a includere la cartella principale del progetto (la root)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mimirbench.evalengines.engines import MemoryEvalEngine
from dotenv import load_dotenv

# Carichi l'ambiente UNA SOLA VOLTA all'avvio del programma
# (Puoi usare un path assoluto o relativo al main)
# PERCORSO ASSOLUTO AL FILE .ENV
CARTELLA_BASE = os.path.dirname(os.path.abspath(__file__))
percorso_env = os.path.join(CARTELLA_BASE, "api_key.env")
load_dotenv(percorso_env)

# Trova la cartella radice dove si trova questo script (main.py)
CARTELLA_BASE = os.path.dirname(os.path.abspath(__file__))

# Percorso del JSON in ingresso
# Per test, il json è stato ridotto a 5 domande
percorso_json = os.path.join(CARTELLA_BASE, "output", "extracted_traces_data_test3.json")

# Percorso del CSV in uscita (esattamente dove lo vuoi tu)
percorso_csv = os.path.join(CARTELLA_BASE, "output", "risultati_test3_Memory.csv")

# Istanzia il motore
engine = MemoryEvalEngine(
    json_dati_path = percorso_json,
    output_csv_path = percorso_csv,
    parallel_launches = 2,
)

# Lanci la valutazione. L'engine prenderà l'ambiente corrente (con la chiave API),
# lo formatterà, e lo inietterà in modo sicuro al test isolato.
engine.run()