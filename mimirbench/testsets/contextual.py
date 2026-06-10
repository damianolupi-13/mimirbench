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
#
# ---
# MODIFICA: Questo file contiene codice originariamente derivato da
# vibrantlabsai/ragas (https://github.com/vibrantlabsai/ragas)
# ed è stato modificato o esteso per le specifiche di MimirBench.

import pandas as pd
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader

# Import necessari Ragas
from ragas.llms import llm_factory
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.embeddings import OpenAIEmbeddings
from ragas.testset import TestsetGenerator
from ragas.utils import num_tokens_from_string

# Importiamo TUTTI i transform del default + NodeType per i filtri
from ragas.testset.graph import NodeType
from ragas.testset.transforms.extractors import (
    HeadlinesExtractor,
    SummaryExtractor,
    EmbeddingExtractor,
    TopicDescriptionExtractor,  # Sostituisce ThemesExtractor nella stable
    NERExtractor
)

from ragas.testset.transforms import (
    HeadlineSplitter,
    CosineSimilarityBuilder,
    CustomNodeFilter,
    OverlapScoreBuilder,
    Parallel
)

from mimirbench.testsets.base import BaseTestset

#Implementazione classe
class ContextualTestset(BaseTestset):
    """
        Classe per caricare documenti aziendali (PDF, DOCX, TXT)
        e prepararli per la generazione testset di RAGAS.

        Args:
            testset_size (int): Il numero totale di domande/casi di test che RAGAS dovrà generare
            model (str): Il nome o l'ID del modello LLM da utilizzare per la generazione (es. "gpt-4o", "gpt-5-nano")
            client (Any): L'istanza del client API per connettersi al modello linguistico (es. l'oggetto restituito da `AsyncOpenAI()`)
            embedding (str): Il nome o l'ID del modello di embedding da utilizzare per la vettorializzazione dei chunk (es. "text-embedding-3-small")
            context (str): Le direttive di sistema (system prompt) fornite al generatore.
                           Utile per forzare la lingua dell'output e impostare regole restrittive
                           sulla struttura delle domande
            max_tokens (int, opzionale): Il limite massimo di token gestibili dal modello LLM per una singola richiesta. Default: 8192
            soglia_limite (int, opzionale): La soglia massima di chunk/pagine tollerata in memoria.
                                            Se il documento caricato supera questo limite, viene
                                            attivato un meccanismo di sicurezza (taglio). Default: 100
            pagine_da_estrarre (int, opzionale): Se viene superata la `soglia_limite`, indica la quantità
                                                 di chunk/pagine da mantenere dopo il taglio. Default: 100
            pagina_di_partenza (int, opzionale): L'indice della pagina o del chunk da cui far partire
                                                 il taglio di sicurezza. Default: 1.
    """

    def __init__(self, testset_size: int, model: str, client, embedding: str, context: str, provider: str = "openai", max_tokens=8192,
                 soglia_limite=100, pagine_da_estrarre=100, pagina_di_partenza=1):
        super().__init__()
        # Parametri di configurazione (Protected)
        self._testset_size = testset_size
        self._llm_model = model
        self._model_client = client
        self._embedding_model = embedding
        self._llm_generator_context = context
        self._provider = provider
        self._max_allowed_tokens = max_tokens
        self._soglia_limite = soglia_limite
        self._pagine_da_estrarre = pagine_da_estrarre
        self._pagina_di_partenza = pagina_di_partenza

    def load(self, filepath: str):
        if filepath not in self._loaded_filepaths:
            self._loaded_filepaths.append(filepath)
            extension = Path(filepath).suffix.lower()
            nuovi_docs = []

            match extension:
                case ".pdf":
                    loader = PyPDFLoader(filepath)
                    nuovi_docs = loader.load()
                case ".docx":
                    loader = UnstructuredWordDocumentLoader(filepath)
                    nuovi_docs = loader.load()
                case ".txt":
                    loader = TextLoader(filepath, encoding="utf-8")
                    nuovi_docs = loader.load()
                case ".md":
                    loader = UnstructuredMarkdownLoader(filepath)
                    nuovi_docs = loader.load()
                case ".csv" | ".xlsx":
                    if extension == ".csv":
                        df = pd.read_csv(filepath)
                    else:
                        df = pd.read_excel(filepath, engine="openpyxl")
                    documenti_tabellari = []
                    righe_per_doc = 10
                    for i in range(0, len(df), righe_per_doc):
                        batch = df.iloc[i: i + righe_per_doc]
                        testo_batch = ""
                        for indice, riga in batch.iterrows():
                            testo_riga = "; ".join([f"{col}: {val}" for col, val in riga.items() if pd.notna(val)])
                            testo_batch += f"Record {indice}: {testo_riga}\n"
                        doc = Document(page_content=testo_batch, metadata={"source": filepath, "row_index": i})
                        documenti_tabellari.append(doc)
                    nuovi_docs = documenti_tabellari
                    print(f"Tabella caricata: serializzate {len(nuovi_docs)} righe")
                case _:
                    raise ValueError(f"Formato file non supportato: {extension}")

            print(f"\n[i] File '{filepath}' ({len(nuovi_docs)} chunks) aggiunto")

            if len(nuovi_docs) > self._soglia_limite:
                print(f"\n[!] ALERT: Rilevati {len(nuovi_docs)} chunks. Il documento è molto denso.")
                start_page = self._pagina_di_partenza
                end_page = min(start_page + self._pagine_da_estrarre, len(nuovi_docs))
                nuovi_docs = nuovi_docs[start_page:end_page]
                print(f"[!] TAGLIO APPLICATO: Mantenuti {len(nuovi_docs)} chunks (da {start_page} a {end_page}).")
            else:
                print(f"\n[i] Documento completo in elaborazione.")

            self._docs.extend(nuovi_docs)
            print(f"\nAccodamento completato. Chunks totali in memoria pronti per il testset: {len(self._docs)}")
        else:
            print("File già caricato!")

    def generate_testset(self, output_csv_path: str):
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_rows', None)

        target_path = Path(output_csv_path)
        if target_path.suffix.lower() != '.csv':
            target_path = target_path.with_suffix('.csv')

        self._output_csv = str(target_path)

        match self._provider:
            case "openai":
                generator_llm = llm_factory(self._llm_model, client=self._model_client, max_tokens=self._max_allowed_tokens)
                generator_embeddings = OpenAIEmbeddings(client=self._model_client, model=self._embedding_model)

            case "anthropic":
                generator_llm = ChatAnthropic(model_name=self._llm_model, max_tokens_to_sample=self._max_allowed_tokens,timeout=None, stop=None)
                generator_embeddings = OpenAIEmbeddings(client=self._model_client, model=self._embedding_model)

            case "google" | "gemini":
                generator_llm = ChatGoogleGenerativeAI(model=self._llm_model, max_output_tokens=self._max_allowed_tokens)
                generator_embeddings = GoogleGenerativeAIEmbeddings(model=self._embedding_model)

            case _:
                raise ValueError(f"[!] Provider '{self._provider}' non supportato per la generazione del testset.")

        counts = [num_tokens_from_string(doc.page_content) for doc in self._docs]
        pct_long = sum(1 for c in counts if c > 500) / len(self._docs)
        pct_med = sum(1 for c in counts if 101 <= c <= 500) / len(self._docs)

        def filter_doc_long(n):
            return n.type == NodeType.DOCUMENT and num_tokens_from_string(n.properties.get("page_content", "")) > 500

        def filter_doc_med(n):
            return n.type == NodeType.DOCUMENT and num_tokens_from_string(n.properties.get("page_content", "")) > 100

        def filter_chunks(n):
            return n.type == NodeType.CHUNK

        custom_transforms = []
        if pct_long >= 0.25:
            print(f"\n>>> Rilevati DOCUMENTI LUNGHI ({pct_long:.0%}).")
            custom_transforms = [
                HeadlinesExtractor(llm=generator_llm, filter_nodes=filter_doc_long),
                HeadlineSplitter(filter_nodes=lambda n: 'headlines' in n.properties and n.properties['headlines']),
                SummaryExtractor(llm=generator_llm, filter_nodes=filter_doc_long),
                CustomNodeFilter(llm=generator_llm, filter_nodes=filter_chunks),
                Parallel(
                    EmbeddingExtractor(embedding_model=generator_embeddings, property_name="summary_embedding",
                                       embed_property_name="summary", filter_nodes=filter_doc_long),
                    TopicDescriptionExtractor(llm=generator_llm, filter_nodes=filter_chunks),
                    NERExtractor(llm=generator_llm, filter_nodes=filter_chunks)
                ),
                Parallel(
                    CosineSimilarityBuilder(property_name="summary_embedding", new_property_name="summary_similarity",
                                            threshold=0.7, filter_nodes=filter_doc_long),
                    OverlapScoreBuilder(threshold=0.01, filter_nodes=filter_chunks)
                )
            ]
        elif pct_med >= 0.25:
            print(f"\n>>> Rilevati DOCUMENTI MEDI ({pct_med:.0%}).")
            custom_transforms = [
                SummaryExtractor(llm=generator_llm, filter_nodes=filter_doc_med),
                CustomNodeFilter(llm=generator_llm),
                Parallel(
                    EmbeddingExtractor(embedding_model=generator_embeddings, property_name="summary_embedding",
                                       embed_property_name="summary", filter_nodes=filter_doc_med),
                    TopicDescriptionExtractor(llm=generator_llm, filter_nodes=filter_doc_med),
                    NERExtractor(llm=generator_llm)
                ),
                Parallel(
                    CosineSimilarityBuilder(property_name="summary_embedding", new_property_name="summary_similarity",
                                            threshold=0.5, filter_nodes=filter_doc_med),
                    OverlapScoreBuilder(threshold=0.01)
                )
            ]
        else:
            print("\n>>> Documenti molto corti. Inserire documenti più lunghi.")

        generator = TestsetGenerator(
            llm=generator_llm,
            embedding_model=generator_embeddings,
            llm_context=self._llm_generator_context
        )

        print("\nAvvio generazione...")
        dataset = generator.generate_with_langchain_docs(
            self._docs,
            testset_size=self._testset_size,
            transforms=custom_transforms,
        )

        df_completo = dataset.to_pandas()
        df_domande = df_completo[['user_input']]
        df_domande.to_csv(self._output_csv, index=False)
        print("\n=== GENERAZIONE COMPLETATA ===")
        for i, domanda in enumerate(df_domande['user_input']):
            print(f"{i}. {domanda}")