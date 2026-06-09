# Script di test del download delle traces da Langfuse
from mimirbench.tracings import MemoryTraceExtractor

from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv("langfusekeys.env")
langfuse = Langfuse(timeout=60)

test_id_used = "2dfd19eb-ed44-4955-b5cf-67de5891121b"

testLangfuseExtraction = MemoryTraceExtractor(langfuse)

output = "output/extracted_traces_data_test3fittizio.json"

testLangfuseExtraction.extracting(output, test_id_used)