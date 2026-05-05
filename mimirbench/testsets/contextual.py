#Implementare classe testset modulo contestuale

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
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader

# Import necessari Ragas
from ragas.llms import llm_factory
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
            spiegare gli argomenti
    """
    def __init__(self, testset_size: int, model: str, client, embedding: str, context: str, language: str, max_tokens = 8192,
                 soglia_limite=100, pagine_da_estrarre=100, pagina_di_partenza=1):
        super().__init__()
        self.testset_size = testset_size
        self.llm_model = model
        self.model_client = client
        self.embedding_model = embedding
        self.llm_generator_context = context
        self.language = language
        self.max_allowed_tokens = max_tokens
        self.soglia_limite = soglia_limite
        self.pagine_da_estrarre = pagine_da_estrarre
        self.pagina_di_partenza = pagina_di_partenza

    #Implementazione metodo load del caso base per caso contextual
    def load(self,  filepath: str):
        # Salviamo i file caricati in una lista
        if not filepath in self.loaded_filepaths:
            self.loaded_filepaths.append(filepath)

            extension = Path(filepath).suffix.lower()

            #Match/Case basato sull'extension
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
                case _:
                    raise ValueError(f"Formato file non supportato: {extension}")

            #Eventualmente inserire un taglio del documento da versioni passati
            print(f"\n[i] File '{filepath}' ({len(nuovi_docs)} chunks) aggiunto")

            # Accodiamo i docs nuovi all'insieme
            self.docs.extend(nuovi_docs)
            print(f"\nAccodamento completato. Chunks totali in memoria pronti per il testset: {len(self.docs)}")
        else:
            print("File già caricato!")

    def generate_testset(self, output_csv_path: str):
        # Valori di settings
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_rows', None)

        self.output_csv = output_csv_path

        #Implementazione metodo di estrazione e costruzione grafo per il testset di Ragas
        generator_llm = llm_factory(self.llm_model, client=self.model_client, max_tokens=self.max_allowed_tokens)
        generator_embeddings = OpenAIEmbeddings(client=self.model_client, model=self.embedding_model)

        # --- FUNZIONI DI FILTRO PER I COMPONENTI "LOCALI" ---
        #Contiamo e facciamo la media dei tipi di chunks estratti dai documenti (in base al conteggio dei token)
        counts = [num_tokens_from_string(doc.page_content) for doc in self.docs]
        pct_long = sum(1 for c in counts if c > 500) / len(self.docs)
        pct_med = sum(1 for c in counts if 101 <= c <= 500) / len(self.docs)

        # Funzioni di filtro per smistare il traffico
        def filter_doc_long(n):
            return n.type == NodeType.DOCUMENT and num_tokens_from_string(n.properties.get("page_content", "")) > 500

        def filter_doc_med(n):
            return n.type == NodeType.DOCUMENT and num_tokens_from_string(n.properties.get("page_content", "")) > 100

        def filter_chunks(n):
            return n.type == NodeType.CHUNK

        # 3. LOGICA DINAMICA (Replicata dal default_transforms di Ragas)
        custom_transforms = []

        # --- BIVIO LOGICO --- Costruzione dei transforms per il grafo di Ragas, a seconda della lunghezza dei chunks del documento
        if pct_long >= 0.25:
            print(f"\n>>> Rilevati DOCUMENTI LUNGHI ({pct_long:.0%}). Applico Splitter.")
            custom_transforms = [
                HeadlinesExtractor(llm=generator_llm, filter_nodes=filter_doc_long),
                # FIX: Non crasha se mancano le headlines
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
                    # Soglia abbassata per Multi-hop
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
            print("\n>>> Documenti molto corti. Inserire documenti pù lunghi.")

        # Generator
        lingua_scelta = self.language
        generator = TestsetGenerator(
            llm=generator_llm,
            embedding_model=generator_embeddings,
            llm_context=f"IMPORTANTISSIMO: Devi generare i nomi delle Personas, le descrizioni, gli scenari e le domande finali ESCLUSIVAMENTE in questa lingua: {lingua_scelta}."
        )

        # Risultato
        print("\nAvvio generazione...")
        # Passiamo 'transforms=custom_transforms' per attivare tutto
        dataset = generator.generate_with_langchain_docs(
            self.docs,
            testset_size=self.testset_size,
            transforms=custom_transforms,
        )

        df_completo = dataset.to_pandas()
        df_domande = df_completo[['user_input']]
        df_domande.to_csv(self.output_csv, index=False)
        print("\n=== GENERAZIONE COMPLETATA ===")
        for i, domanda in enumerate(df_domande['user_input']):
            print(f"{i}. {domanda}")