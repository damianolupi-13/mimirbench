#Implementare classe BaseTestset astratta estendibile

from abc import ABC, abstractmethod

class BaseTestset(ABC):
    """
        Classe testset base astratta. Definisce oggetti testset e genera file .CSV senza particolari proprietà.
        Da estendere per creare un testset particolare.
    """

    #Costruttore con filepaths dei dati/documenti da analizzare o da cui prendere i testset di domande
    def __init__(self):
        self.loaded_filepaths = []
        self.docs = []
        self.output_csv = None

    #Metodo astratto per il caricamento dei dati/documenti
    @abstractmethod
    def load(self,  filepath: str):
        """Ogni sottoclasse deve implementare come caricare i dati/documenti."""
        pass

    #Metodo astratto per la generazione del file .CSV testset
    @abstractmethod
    def generate_testset(self, output_csv_path: str):
        """Ogni sottoclasse deve implementare come generare il testset in file .CSV."""
        pass