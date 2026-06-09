# Script di test del download delle traces da Langfuse
from mimirbench.tracings import TechnicalTraceExtractor

from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv("langfusekeys.env")
langfuse = Langfuse(timeout=60)

test_id_used = "0b88abd2-3982-4506-97fa-966a782b64e5"

testLangfuseExtraction = TechnicalTraceExtractor(langfuse)

output = "output/extracted_traces_data_test4.json"

testLangfuseExtraction.extracting(output, test_id_used)