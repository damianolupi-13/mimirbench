#Implementare classe testset modulo operazionale per retenzione memoria

import pandas as pd
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredMarkdownLoader
from mimirbench.testsets.base import BaseTestset

class MemoryTestset(BaseTestset):
    """
        Classe per caricare documenti testuali o tabellari contenenti un copione di domande,
        per convertirle in un testset .CSV in modo tale da valutare la memoria dell'agente.

    Args:
        separator (str) (optional): La stringa usata nel documento TESTUALE per separare una domanda dall'altra.
                         Default: "\n"
    """

    def __init__(self, separator: str = "\n"):
        super().__init__()
        self.separator = separator

    def load(self, filepath: str):
        """Carica il documento e ne estrae il contenuto grezzo."""
        self.data_filepath = filepath
        extension = Path(self.data_filepath).suffix.lower()

        # Match/Case basato sull'estensione
        match extension:
            #Documenti NON Tabellari e Testuali
            case ".pdf":
                loader = PyPDFLoader(self.data_filepath)
                self.docs = loader.load()
            case ".docx":
                loader = Docx2txtLoader(self.data_filepath)
                self.docs = loader.load()
            case ".txt":
                loader = TextLoader(self.data_filepath, encoding="utf-8")
                self.docs = loader.load()
            case ".md":
                loader = UnstructuredMarkdownLoader(self.data_filepath)
                self.docs = loader.load()

            #Documenti Tabellari
            case ".csv":
                self.docs = pd.read_csv(self.data_filepath)
            case ".xlsx":
                self.docs = pd.read_excel(self.data_filepath)

            #Failsafe
            case _:
                raise ValueError(f"Formato file non supportato: {extension}")

        #Controlla se docs è un dataframe o un insieme di documenti NON tabellari
        if isinstance(self.docs, pd.DataFrame):
            print(f"✅ Tabella caricata: lette {len(self.docs)} righe.")
        else:
            print(f"✅ Documento di testo caricato: lette {len(self.docs)} pagine/blocchi.")

    def generate_testset(self, output_csv_path: str):
        """
        Estrae le domande, le normalizza in una singola colonna e salva il testset in CSV.
        """
        if self.docs is None:
            raise ValueError("Nessun documento caricato. Chiama load() prima.")

        domande_pulite = []

        # --- File Tabellari ---
        if isinstance(self.docs, pd.DataFrame):
            df:pd.DataFrame = self.docs
            # Cerca automaticamente la colonna giusta
            if 'user_input' in df.columns:
                domande_pulite = df['user_input'].dropna().astype(str).tolist() #dropna = drop celle vuote (not available)
            elif 'domanda' in df.columns:
                domande_pulite = df['domanda'].dropna().astype(str).tolist()
            else:
                # Fallback: prende la primissima colonna disponibile
                prima_colonna = df.columns[0]
                domande_pulite = df[prima_colonna].dropna().astype(str).tolist()

        # --- File Testuali con Separatore ---
        else:
            # 1. Uniamo tutto il testo
            testo_completo = "\n".join([doc.page_content for doc in self.docs])

            # 2. Affettiamo col separatore scelto dall'utente
            porzioni = testo_completo.split(self.separator)

            # 3. Pulizia spazi e invii accidentali
            for porzione in porzioni:
                domanda = porzione.strip()
                if domanda:
                    domande_pulite.append(domanda)

        # Controllo di sicurezza
        if not domande_pulite:
            print("[!] Attenzione: Nessuna domanda estratta. Controlla il file o il separatore.")
            return

        # Salvataggio Universale
        df_memoria = pd.DataFrame(domande_pulite, columns=["user_input"])
        df_memoria.to_csv(output_csv_path, index=False)

        print(f"Generazione completata! Estratte {len(df_memoria)} domande sequenziali.")
        print(f"File salvato in: {output_csv_path}\n")

        # Anteprima a schermo
        for i, dom in enumerate(domande_pulite[:3], 1):
            print(f" {i}. {dom[:70]}..." if len(dom) > 70 else f" {i}. {dom}")
        if len(domande_pulite) > 3:
            print(" ...")
