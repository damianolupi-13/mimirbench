#Implementare classe BaseTraceExtractor astratta estendibile

from langfuse import Langfuse
from abc import ABC, abstractmethod

class BaseTraceExtractor(ABC):
    """
        Classe trace_extractor base astratta. Definisce oggetti extractor per estrarre tracce da Langfuse.
        Da estendere per creare un extractor particolare.
        È necessario inserire .ENV api key di Langfuse nel proprio script
        per far funzionare gli oggetti della classe.
    """

    #Costruttore con tag delle traces da cercare
    def __init__(self, tracing_tag: str):
        self.tracing_tag = tracing_tag
        self.langfuse_instance = Langfuse()

    #Metodo astratto per il recupero di informazioni necessarie (se serve)
    @abstractmethod
    def fetching(self,  trace_output):
        """Ogni sottoclasse deve implementare se e come recuperare singole informazioni necessarie
           da output specifici della traccia Langfuse."""
        pass

    #Metodo astratto per la generazione del file .JSON
    @abstractmethod
    def extracting(self, output_json_path: str):
        """Ogni sottoclasse deve implementare come estrarre informazioni volute dalle traces Langfuse
           e organizzarle in un JSON."""
        pass