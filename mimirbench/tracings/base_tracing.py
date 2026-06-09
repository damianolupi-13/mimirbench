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

class BaseTraceExtractor(ABC):
    """
        Classe trace_extractor base astratta. Definisce oggetti extractor per estrarre tracce da Langfuse.
        Da estendere per creare un extractor particolare.
        È necessario inserire .ENV api key di Langfuse nel proprio script
        per far funzionare gli oggetti della classe.
    """

    #Costruttore con tag delle traces da cercare
    def __init__(self, langfuse_instance):
        self.langfuse_instance = langfuse_instance

    #Metodo astratto per il recupero di informazioni necessarie (se serve)
    @abstractmethod
    def fetching(self,  trace_output):
        """Ogni sottoclasse deve implementare se e come recuperare singole informazioni necessarie
           da output specifici della traccia Langfuse."""
        pass

    #Metodo astratto per la generazione del file .JSON
    @abstractmethod
    def extracting(self, output_json_path: str, test_id: str):
        """Ogni sottoclasse deve implementare come estrarre informazioni volute dalle traces Langfuse
           e organizzarle in un JSON."""
        pass