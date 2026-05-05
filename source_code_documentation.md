# MimirBench: Documentazione Architetturale

MimirBench è un framework modulare progettato per la valutazione automatizzata di agenti LLM e sistemi RAG (Retrieval-Augmented Generation). L'architettura è suddivisa in tre moduli principali che gestiscono l'intero ciclo di vita del test: preparazione dei dati, osservabilità ed estrazione delle tracce, ed esecuzione delle metriche di valutazione.

Di seguito è riportata la documentazione tecnica dei componenti principali.

---

## 1. Modulo `testsets` (Ingestione e Preparazione Dati)
Questo modulo gestisce il caricamento dei documenti e la generazione dei dataset di test (in formato CSV) utilizzati per valutare l'agente.

*   **`BaseTestset`** (`base.py`):
    *   È la classe astratta principale che definisce l'interfaccia standard per i testset[cite: 3].
    *   Gestisce lo stato interno inizializzando liste per i percorsi dei file caricati (`loaded_filepaths`) e per i documenti elaborati (`docs`)[cite: 3].
    *   Richiede l'implementazione dei metodi `load()` e `generate_testset()` nelle classi figlie[cite: 3].

*   **`ContextualTestset`** (`contextual.py`):
    *   Progettata per la valutazione dei sistemi RAG, supporta il caricamento di file PDF, DOCX, TXT e MD[cite: 4].
    *   Utilizza il motore di Ragas per generare dinamicamente domande sintetiche basate sul contesto aziendale fornito[cite: 4].
    *   Implementa una logica di trasformazione dinamica (es. `HeadlineSplitter`, `CosineSimilarityBuilder`) che adatta il parsing dei nodi in base alla lunghezza media dei documenti elaborati[cite: 4].

*   **`MemoryTechnicalTestset`** (`memory_and_tech.py`):
    *   Progettata per estrarre set di domande predefinite utili a valutare la memoria o aspetti operativi dell'agente[cite: 5].
    *   Supporta l'estrazione da documenti tabellari (CSV, Excel) isolando colonne specifiche (come `user_input` o `domanda`)[cite: 5].
    *   Supporta documenti testuali semplici, suddividendo le domande tramite un separatore personalizzato (es. `\n`) e normalizzandole in un CSV a colonna singola[cite: 5].

---

## 2. Modulo `tracings` (Osservabilità ed Estrazione da Langfuse)
Questo modulo si interfaccia con le API di Langfuse per recuperare e filtrare i dati di esecuzione dell'agente, trasformandoli in file JSON strutturati pronti per la valutazione.

*   **`BaseTraceExtractor`** (`base_tracing.py`):
    *   Classe astratta di base per l'estrazione delle tracce[cite: 6].
    *   Inizializza l'istanza di connessione a Langfuse e definisce i metodi astratti `fetching()` ed `extracting()` che le sottoclassi devono implementare[cite: 6].

*   **`ContextualTraceExtractor`** (`contextual_tracing.py`):
    *   Estrae i dati necessari per i test RAG (valutazione contestuale)[cite: 7].
    *   Recupera l'ultimo stato del nodo `react_agent` per estrarre la domanda dell'utente, la risposta finale dell'assistente e i documenti recuperati nel contesto (`retrieval_context`)[cite: 7].

*   **`MemoryTraceExtractor`** (`memory_tracing.py`):
    *   Specializzata per i test di ritenzione mnemonica[cite: 8].
    *   Analizza e ordina cronologicamente le tracce per identificare la conversazione più lunga[cite: 8].
    *   Ricostruisce lo storico dei turni (`input` e `actual_output`) utilizzando un parser interno che filtra e ignora le chiamate ai tool[cite: 8].

*   **`TechnicalTraceExtractor`** (`tech_tracing.py`):
    *   Estrae i dati operativi e le metriche tecniche dell'agente[cite: 9].
    *   Analizza l'intera traccia per calcolare la latenza totale, contare i token effettivi utilizzati (es. dal nodo `ChatVertexAI`) e registrare quali tool sono stati chiamati dall'agente per rispondere alla richiesta[cite: 9].

---

## 3. Modulo `evalengines` (Motore di Valutazione)
Questo modulo esegue la valutazione formale utilizzando il framework DeepEval. Legge i JSON prodotti dal modulo `tracings` e restituisce i risultati analitici.

*   **`BaseEvalEngine`** (`engines.py`):
    *   Gestisce l'esecuzione del test isolando l'ambiente Python tramite la clonazione di `os.environ`[cite: 2].
    *   Disabilita la telemetria di DeepEval e passa le configurazioni necessarie (come percorsi file e modalità di test) tramite variabili d'ambiente[cite: 2].
    *   Avvia lo script di test tramite un processo secondario (`subprocess.run`)[cite: 2].

*   **Classi Specifiche di Esecuzione** (`engines.py`):
    *   `RagEvalEngine`, `AgentEvalEngine` e `MemoryEvalEngine` estendono il motore di base[cite: 2].
    *   Ciascuna classe imposta automaticamente il flag `eval_mode` corretto (rispettivamente `RAG`, `AGENT`, `MEMORY`) per avviare la suite di test appropriata[cite: 2].

*   **`_hidden_pytest.py`**:
    *   Lo script interno che esegue le misurazioni di test[cite: 1].
    *   Contiene l'implementazione delle metriche custom di MimirBench, tra cui `ToolHallucinationMetric`, `LatencyMetric` e `TokenEfficiencyMetric`[cite: 1].
    *   Mappa dinamicamente i dati JSON all'interno degli oggetti formali di DeepEval (`LLMTestCase` o `ConversationalTestCase`) in base alla modalità di valutazione selezionata[cite: 1].

### Keypoints di Utilizzo degli Engine
*   **Uso Standard:** Istanzia un motore specifico (es. `RagEvalEngine` o `AgentEvalEngine`) fornendo il percorso del JSON in input e del CSV di output, quindi chiama `run_evaluations()`[cite: 10]. Non è richiesta configurazione aggiuntiva[cite: 10].
*   **Metriche Personalizzate:** È possibile iniettare metriche custom passando una stringa di entrypoint (es. `"mio_file:mia_funzione"`) al parametro `metrics_provider`[cite: 10].
*   **Modalità Custom:** Per valutare casi d'uso non standard, si può utilizzare `BaseEvalEngine` fornendo un `testcase_builder` personalizzato e invocando `run(eval_mode="NOME_CUSTOM")`[cite: 10].
*   **Controllo Totale:** Tramite la funzione `eject_test_script()`, è possibile estrarre lo script core e passarlo all'engine tramite il parametro `custom_test_script`, alterando profondamente la logica di test senza modificare la libreria[cite: 10].

---

## 4. Modulo `printers` (da completare)
Questo modulo conterrà degli script che permettono di visualizzare i risultati dei test

---

## Script Utility: `utils.py`

*   **`eject_test_script()`**:
    *   Funzione di utilità che permette di esportare lo script Pytest interno (`_hidden_pytest.py`) copiandolo nella directory di lavoro dell'utente.
    *   È progettata per fornire flessibilità, consentendo agli sviluppatori di estendere il core del test o aggiungere metriche personalizzate senza dover modificare direttamente il codice sorgente del framework.

---

## Quickstart: Come usare la libreria
Per eseguire un ciclo di test completo con MimirBench, le istanziazioni seguono un flusso logico lineare diviso in tre passaggi:

1.  **Preparazione Dati:** Istanzia una classe dal modulo `testsets` (es. `ContextualTestset()`) per analizzare i tuoi documenti e generare un file CSV contenente le domande di test.
2.  **Estrazione Tracce:** Dopo aver fatto rispondere il tuo LLM alle domande e aver inviato i log a Langfuse, istanzia un estrattore dal modulo `tracings` (es. `ContextualTraceExtractor()`) fornendo il test ID per ottenere un JSON pulito della conversazione.
3.  **Valutazione Finale:** Istanzia l'engine di riferimento dal modulo `evalengines` (es. `RagEvalEngine(json_dati_path="...", output_csv_path="...")`) e invoca `.run_evaluations()` per calcolare i punteggi finali e salvarli in CSV[cite: 10].

---

## Script Non Utilizzati (Di Prova / Da Rimuovere)
I seguenti file sono script di sandbox e test isolati, utilizzati durante la fase di sviluppo iniziale per validare le integrazioni con framework esterni (DeepEval, Ragas, Langfuse). Non fanno parte dell'architettura *core* di MimirBench e sono destinati a essere rimossi per mantenere il repository pulito e ottimizzato.

**Directory principale**
*   `forTaggingTraces.py`

**Modulo `evalengines`**
*   `test_DeepEval1.py`
*   `test_DeepEval_memory_metric.py`
*   `test_DeepEval_technical_agent_metrics.py`

**Modulo `testsets`**
*   `provaRagas4.py`

**Modulo `tracings`**
*   `provaLangfuse1.py`
*   `provaLangfuse_for_memory.py`
*   `provaLangfuse_for_technical_agent.py`