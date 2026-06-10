from mimirbench.printers.pdf_printer import MimirPDFPrinter
from mimirbench.printers.langfuse_printer import LangfusePrinter

from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv("langfusekeys.env")

# 1. Stampa in PDF
pdf_maker = MimirPDFPrinter(csv_file_path="output/risultati_testRAG1.csv", output_pdf_path="output/Report_test2.pdf")
pdf_maker.genera_report()

# 2. Esporta su Langfuse
# (Assumi che 'langfuse_client' sia la tua istanza langfuse = Langfuse(...))
langfuse_client = Langfuse()
langfuse_exporter = LangfusePrinter(csv_file_path="output/risultati_testRAG1.csv", langfuse_client=langfuse_client)
langfuse_exporter.push_scores()