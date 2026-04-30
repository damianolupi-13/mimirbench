import shutil
import os


def eject_test_script(destinazione="custom_mimir_test.py"):
    """
    Estrae lo script Pytest interno di MimirBench copiandolo nella directory di lavoro.
    Ideale per chi ha bisogno di modificare il core del test.
    """
    cartella_corrente = os.path.dirname(os.path.abspath(__file__))
    script_originale = os.path.join(cartella_corrente, "evaluations", "_hidden_pytest.py")

    shutil.copy(script_originale, destinazione)
    print(f"Script di test esportato con successo in: {destinazione}")
    print("Per utilizzarlo, passalo all'Engine tramite il parametro 'custom_test_script'.")