# Copyright 2026 Damiano Lupi (https://github.com/damianolupi-13)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod

class BaseTestset(ABC):
    """
        Classe testset base astratta. Definisce oggetti testset e genera file .CSV senza particolari proprietà.
        Da estendere per creare un testset particolare.
    """

    #Costruttore con filepaths dei dati/documenti da analizzare o da cui prendere i testset di domande
    def __init__(self):
        self._loaded_filepaths = []
        self._docs = []
        self._output_csv = None

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