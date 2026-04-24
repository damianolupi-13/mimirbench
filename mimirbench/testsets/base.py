#Implementare classe BaseTestset astratta estendibile

from abc import ABC, abstractmethod

class BaseTestset(ABC):
    """
        Classe testset base astratta. Definisce oggetti testset e genera file .CSV senza particolari proprietà.
        Da estendere per creare un testset particolare.
    """

    #Costruttore con filepath dei dati/documenti da analizzare o da cui prendere il testset di domande
    def __init__(self, filepath: str, output_csv_path: str):
        self.data_filepath = filepath
        self.docs = None
        self.output_csv = output_csv_path

    #Metodo astratto per il caricamento dei dati/documenti
    @abstractmethod
    def load(self):
        """Ogni sottoclasse deve implementare come caricare i dati/documenti."""
        pass

    #Metodo astratto per la generazione del file .CSV testset
    @abstractmethod
    def generate_testset(self):
        """Ogni sottoclasse deve implementare come generare il testset in file .CSV."""
        pass