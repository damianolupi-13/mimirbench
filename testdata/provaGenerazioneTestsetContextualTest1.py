from mimirbench.testsets.contextual import ContextualTestset
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv("api_key.env")
os.environ["OPENAI_API_KEY"] = os.environ.get("API_KEY")

#Arguments
test_size = 25
model = "gpt-5-nano"
client = AsyncOpenAI()
embeddings = "text-embedding-3-small"
lingua_scelta = "Italiano"
context = f"IMPORTANTISSIMO: Devi generare i nomi delle Personas, le descrizioni, gli scenari e le domande finali ESCLUSIVAMENTE in questa lingua: {lingua_scelta}."
max_tokens = 8192

#Costruttore test
test = ContextualTestset(test_size, model, client, embeddings, context, max_tokens, soglia_limite=20, pagine_da_estrarre=20)

#Esecuzione test
input_data = "input/relazione-finanziaria-annuale-ENI-2023-prova3.pdf"
test.load(input_data)
output_path = "output/testset.csv"
test.generate_testset(output_path)

print("\n\nESECUZIONE TEST <GENERAZIONE TESTSET CONTESTUALE> COMPLETATO CON SUCCESSO!")


