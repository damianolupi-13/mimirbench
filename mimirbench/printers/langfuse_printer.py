import pandas as pd
import time
import os

class LangfusePrinter:
    """Classe per esportare i risultati su Langfuse usando le intestazioni esatte del CSV."""

    def __init__(self, csv_file_path: str, langfuse_client):
        # Attributi protetti ad uso interno
        self._csv_file = csv_file_path
        self._langfuse = langfuse_client

    def push_scores(self):
        """Metodo pubblico per l'avvio della sincronizzazione"""
        if not os.path.exists(self._csv_file):
            print(f"[Langfuse Printer] ERRORE: File {self._csv_file} non trovato.")
            return

        try:
            df = pd.read_csv(self._csv_file, sep=";")
            df.columns = df.columns.str.strip()
        except Exception as e:
            print(f"[Langfuse Printer] ERRORE nella lettura del CSV: {e}")
            return

        if "Trace_ID" not in df.columns:
            print(f"[Langfuse Printer] ERRORE CRITICO: Colonna 'Trace_ID' non trovata. Colonne lette: {list(df.columns)}")
            return

        # Trova colonne dinamicamente
        eval_score = next((c for c in df.columns if "punteggio" in c.lower() or "score" in c.lower()), None)
        colonna_commento = next((c for c in df.columns if "ragionamento" in c.lower() or "motiva" in c.lower() or "reason" in c.lower()), None)
        colonna_metrica = next((c for c in df.columns if "metrica" in c.lower() or "metric" in c.lower()), None)

        if not eval_score:
            print("[Langfuse Printer] ERRORE: Nessuna colonna relativa al punteggio trovata nell'intestazione.")
            return

        df_validi = df[df["Trace_ID"].notna() & (df["Trace_ID"] != "")]

        if df_validi.empty:
            print("[Langfuse Printer] Nessuna traccia valida da sincronizzare.")
            return

        print(f"[Langfuse Printer] Sincronizzazione in corso...")
        if colonna_metrica:
            print(f"  -> Colonna metrica rilevata ('{colonna_metrica}').")
        else:
            print(f"  -> Nessuna colonna metrica. Nome Score di default su Langfuse: '{eval_score}'")

        successi = 0
        fallimenti = 0

        for index, row in df_validi.iterrows():
            trace_id = str(row["Trace_ID"]).strip()
            punteggio = row[eval_score]

            if pd.isna(punteggio):
                continue

            motivazione = str(row[colonna_commento]) if colonna_commento and pd.notna(row[colonna_commento]) else ""
            nome_score_finale = str(row[colonna_metrica]) if colonna_metrica and pd.notna(row[colonna_metrica]) else eval_score

            try:
                self._langfuse.create_score(
                    trace_id=trace_id,
                    name=nome_score_finale,
                    value=float(punteggio),
                    comment=motivazione
                )
                successi += 1
            except Exception as e:
                print(f"   [!] Errore API su Trace {trace_id}: {e}")
                fallimenti += 1

            time.sleep(0.1)

        self._langfuse.flush()

        print("\n=== REPORT SINCRONIZZAZIONE LANGFUSE ===")
        print(f"Metriche caricate con successo: {successi}")
        print(f"Errori di caricamento: {fallimenti}")
        print("========================================")