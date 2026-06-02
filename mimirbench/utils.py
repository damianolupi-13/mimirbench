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

import shutil
import os


def eject_test_script(destinazione="custom_mimir_test.py"):
    """
    Estrae lo script Pytest interno di MimirBench copiandolo nella directory di lavoro.
    Ideale per chi ha bisogno di modificare il core del test.
    """
    cartella_corrente = os.path.dirname(os.path.abspath(__file__))
    script_originale = os.path.join(cartella_corrente, "evaluations", "test_hidden_pytest.py")

    shutil.copy(script_originale, destinazione)
    print(f"Script di test esportato con successo in: {destinazione}")
    print("Per utilizzarlo, passalo all'Engine tramite il parametro 'custom_test_script'.")