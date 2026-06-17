from mimirbench.testsets.contextual import ContextualTestset
from openai import AsyncOpenAI
import os
import httpx
from dotenv import load_dotenv

load_dotenv("api_key.env")
os.environ["OPENAI_API_KEY"] = os.environ.get("API_KEY")

#Arguments
test_size = 25
model = "gpt-5-nano"
#Limitatore del lancio parallelo di richieste ad OpenAI per garantire la correttezza dell'operazione su numeri elevati di richieste
limitatore_connessioni = httpx.AsyncClient(
    limits=httpx.Limits(max_connections=25, max_keepalive_connections=25)
)

client = AsyncOpenAI(
    http_client=limitatore_connessioni,
    max_retries=5  # Se una richiesta fallisce, ritenta fino a 5 volte in automatico
)
embeddings = "text-embedding-3-small"
lingua_scelta = "Italiano"
context = f"IMPORTANTISSIMO: Devi generare i nomi delle Personas, le descrizioni, gli scenari e le domande finali ESCLUSIVAMENTE in questa lingua: {lingua_scelta}."
provider= "openai"
max_tokens = 8192

#Costruttore test
test = ContextualTestset(test_size, model, client, embeddings, context, provider, max_tokens, soglia_limite=50, pagine_da_estrarre=50)

#Esecuzione test
input_data = "input/relazione-finanziaria-annuale-ENI-2023.pdf"
test.load(input_data)
output_path = "output/testset_prova_pre_tesi.csv"
test.generate_testset(output_path)

print("\n\nESECUZIONE TEST <GENERAZIONE TESTSET CONTESTUALE> COMPLETATO CON SUCCESSO!")


