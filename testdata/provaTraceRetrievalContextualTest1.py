# Script di test del download delle traces da Langfuse
from mimirbench.tracings import ContextualTraceExtractor

from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv("langfusekeys.env")
langfuse = Langfuse()

test_id_used = "0b88abd2-3982-4506-97fa-966a782b64e5"

testLangfuseExtraction = ContextualTraceExtractor(langfuse)

output = "output/extracted_traces_data_test1.json"

testLangfuseExtraction.extracting(output, test_id_used)