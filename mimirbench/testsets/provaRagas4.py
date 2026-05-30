import os
from dotenv import load_dotenv
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from openai import AsyncOpenAI

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
    # --- NUOVI COMPONENTI DAL DEFAULT ---
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


pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

load_dotenv("../../testdata/api_key.env")
os.environ["OPENAI_API_KEY"] = os.environ.get("API_KEY")

status = 3
match status:
    case 1:
        #Load pdf
        file_path = "relazione-finanziaria-annuale-ENI-2023-prova3.pdf"
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    case 2:
        #Load docx
        file_path = "prova4.docx"
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
    case 3:
        # Load txt
        file_path = "prova5.txt"
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()

# Parametri di sicurezza generali per qualsiasi documento
SOGLIA_LIMITE = 100
PAGINE_DA_ESTRARRE = 100
PAGINA_DI_PARTENZA = 1

if len(docs) > SOGLIA_LIMITE:
    print(f"\n[!] ALERT: Rilevate {len(docs)} pagine. Il documento è molto denso.")

    start_page = PAGINA_DI_PARTENZA

    # La funzione min() assicura che se si supera la grandezza del PDF,
    # l'end_page si fermerà all'ultima pagina reale del documento.
    end_page = min(start_page + PAGINE_DA_ESTRARRE, len(docs))

    docs = docs[start_page:end_page]

    print(f"[!] TAGLIO APPLICATO: Mantenute {len(docs)} pagine (dalla {start_page} alla {end_page}).")
else:
    print(f"\n[i] Documento di {len(docs)} pagine: entro i limiti di sicurezza, elaboro il file completo.")

openai_client = AsyncOpenAI()
generator_llm = llm_factory("gpt-5-mini", client=openai_client, max_tokens=8192)
generator_embeddings = OpenAIEmbeddings(client=openai_client, model="text-embedding-3-small")


# --- FUNZIONI DI FILTRO PER I COMPONENTI "LOCALI" ---
# I due filtri che ci servono per smistare il traffico
counts = [num_tokens_from_string(doc.page_content) for doc in docs]
pct_long = sum(1 for c in counts if c > 500) / len(docs)
pct_med  = sum(1 for c in counts if 101 <= c <= 500) / len(docs)

# Funzioni di filtro per smistare il traffico
def filter_doc_long(n):
    return n.type == NodeType.DOCUMENT and num_tokens_from_string(n.properties.get("page_content", "")) > 500

def filter_doc_med(n):
    return n.type == NodeType.DOCUMENT and num_tokens_from_string(n.properties.get("page_content", "")) > 100

def filter_chunks(n):
    return n.type == NodeType.CHUNK

# 3. LOGICA DINAMICA (Replicata dal default_transforms di Ragas)
custom_transforms = []

# --- BIVIO LOGICO ---
if pct_long >= 0.25:
    print(f">>> Rilevati DOCUMENTI LUNGHI ({pct_long:.0%}). Applico Splitter.")
    custom_transforms = [
        HeadlinesExtractor(llm=generator_llm, filter_nodes=filter_doc_long),
        # FIX AGID: Non crasha se mancano le headlines
        HeadlineSplitter(filter_nodes=lambda n: 'headlines' in n.properties and n.properties['headlines']),
        SummaryExtractor(llm=generator_llm, filter_nodes=filter_doc_long),
        CustomNodeFilter(llm=generator_llm, filter_nodes=filter_chunks),
        Parallel(
            EmbeddingExtractor(embedding_model=generator_embeddings, property_name="summary_embedding", embed_property_name="summary", filter_nodes=filter_doc_long),
            TopicDescriptionExtractor(llm=generator_llm, filter_nodes=filter_chunks),
            NERExtractor(llm=generator_llm, filter_nodes=filter_chunks)
        ),
        Parallel(
            CosineSimilarityBuilder(property_name="summary_embedding", new_property_name="summary_similarity", threshold=0.7, filter_nodes=filter_doc_long), # Soglia abbassata per Multi-hop
            OverlapScoreBuilder(threshold=0.01, filter_nodes=filter_chunks)
        )
    ]

elif pct_med >= 0.25:
    print(f">>> Rilevati DOCUMENTI MEDI ({pct_med:.0%}). Uso logica Page-level.")
    custom_transforms = [
        # Invertito l'ordine per evitare l'errore "No summary"
        SummaryExtractor(llm=generator_llm, filter_nodes=filter_doc_med),
        CustomNodeFilter(llm=generator_llm),
        Parallel(
            EmbeddingExtractor(embedding_model=generator_embeddings, property_name="summary_embedding", embed_property_name="summary", filter_nodes=filter_doc_med),
            TopicDescriptionExtractor(llm=generator_llm, filter_nodes=filter_doc_med),
            NERExtractor(llm=generator_llm)
        ),
        Parallel(
            CosineSimilarityBuilder(property_name="summary_embedding", new_property_name="summary_similarity", threshold=0.5, filter_nodes=filter_doc_med),
            OverlapScoreBuilder(threshold=0.01)
        )
    ]
else:
    print(">>> Documenti molto corti. Inserire documenti pù lunghi.")

#Generator
lingua_scelta = "Italiano"
generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings,
    llm_context=f"IMPORTANTISSIMO: Devi generare i nomi delle Personas, le descrizioni, gli scenari e le domande finali ESCLUSIVAMENTE in questa lingua: {lingua_scelta}."
)

#Risultato
print("\nAvvio generazione con pipeline Default completa + Fix...")
# Passiamo 'transforms=custom_transforms' per attivare tutto
dataset = generator.generate_with_langchain_docs(
    docs,
    testset_size=25,
    transforms=custom_transforms,
)

df_completo = dataset.to_pandas()
df_domande = df_completo[['user_input']]
df_domande.to_csv("domande_per_mimir_v04_prova6.csv", index=False)
print("\n=== GENERAZIONE COMPLETATA ===")
for i, domanda in enumerate(df_domande['user_input']):
    print(f"{i}. {domanda}")