import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotx
from fpdf import FPDF
import os

CSV_FILE = "Risultati_Mimir_Pytest.csv"
PDF_OUTPUT = "Report_Mimir_Nord_Editorial.pdf"

# ==========================================
# PALETTE COLORI NORD
# ==========================================
COLOR_SUCCESS = '#88C0D0'
COLOR_FAIL = '#BF616A'
COLOR_RADAR = '#A3BE8C'
COLOR_TEXT_MATPLOT = '#3B4252' # Colore Hex per Matplotlib

NORD_DARK_RGB = (46, 52, 64)
NORD_LIGHT_RGB = (236, 239, 244)
NORD_CYAN_RGB = (136, 192, 208)
NORD_TEXT_RGB = (59, 66, 82)
NORD_GREEN_RGB = (143, 188, 143)
NORD_RED_RGB = (191, 97, 106)
NORD_YELLOW_RGB = (235, 203, 139)


def prepara_dati_scalabili(df):
    df_punteggi = df.pivot_table(index="Input", columns="Metrica", values="Punteggio", aggfunc='first').reset_index()
    # Rinomina ogni colonna togliendo "Metric" (se presente)
    df_punteggi.columns = [c.replace("Metric", "") for c in df_punteggi.columns]

    df_tempi = df.groupby("Input")["Durata (s)"].sum().reset_index()
    df_tempi.rename(columns={"Durata (s)": "Tempo_Totale_Domanda_s"}, inplace=True)
    master_df = pd.merge(df_punteggi, df_tempi, on="Input")

    # Calcoliamo lo score medio per ogni riga (domanda)
    cols_score = [c for c in master_df.columns if c not in ["Input", "Tempo_Totale_Domanda_s"]]
    master_df['Score_Medio_Domanda'] = master_df[cols_score].mean(axis=1)

    metriche_cols = [c for c in master_df.columns if
                     c not in ["Input", "Tempo_Totale_Domanda_s", "Score_Medio_Domanda"]]
    return master_df, metriche_cols


# ==========================================
# FASE 1: MOTORE GRAFICO (Generazione PNG)
# ==========================================
def genera_immagini_temporanee(df_raw, df_master, lista_metriche):
    print("📸 Generazione dei grafici in stile Nord in corso...")
    plt.style.use(matplotx.styles.nord)

    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'text.color': '#2E3440',
        'axes.labelcolor': '#2E3440',
        'xtick.color': '#2E3440',
        'ytick.color': '#2E3440',
        'savefig.facecolor': 'white'
    })

    # 1. Ciambella (Migliorata leggibilità)
    test_passed = len(df_raw[df_raw["Esito"] == "PASSED"])
    test_failed = len(df_raw) - test_passed
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    wedges, texts, autotexts = ax.pie(
        [test_passed, test_failed], labels=['PASSED', 'FAILED'],
        colors=[COLOR_SUCCESS, COLOR_FAIL], autopct='%1.1f%%',
        startangle=90, pctdistance=0.80,  # Percentuali più esterne
        wedgeprops=dict(width=0.4, edgecolor='none')
    )
    plt.setp(autotexts, size=11, weight="bold", color="white")
    ax.set_title("Global Success Rate", fontsize=17, fontweight='bold')
    plt.savefig("temp_pie.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 1.2. Top 10 Performance Bar Chart
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    # Ordiniamo per score medio e prendiamo i primi 10
    top_10 = df_master.sort_values(by='Score_Medio_Domanda', ascending=False).head(10).iloc[::-1]

    # Usiamo l'indice originale per etichettare le domande (es. Q1, Q5...)
    labels = [f"Q{i + 1}" for i in top_10.index]
    scores = top_10['Score_Medio_Domanda']

    bars = ax.barh(labels, scores, color=COLOR_RADAR, height=0.7)

    # Aggiunta dei valori numerici (Fix del colore effettuato qui)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{width:.2f}', va='center', fontsize=8, fontweight='bold', color=COLOR_TEXT_MATPLOT)

    ax.set_xlim(0, 1.25)
    ax.set_title("Top 10 Performance by Question", fontsize=20, pad=20, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig("temp_scores.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Radar Chart
    fig = plt.figure(figsize=(5, 5))
    ax_radar = fig.add_subplot(111, polar=True)
    valori_medi = df_raw.groupby("Metrica")["Punteggio"].mean().tolist()
    valori_medi += valori_medi[:1]
    angoli = np.linspace(0, 2 * np.pi, len(lista_metriche), endpoint=False).tolist()
    angoli += angoli[:1]
    ax_radar.set_title("Mimir Stats Chart", fontsize=20, pad=40, fontweight='bold')
    ax_radar.plot(angoli, valori_medi, color=COLOR_RADAR, linewidth=2.5)
    ax_radar.fill(angoli, valori_medi, color=COLOR_RADAR, alpha=0.2)
    ax_radar.set_xticks(angoli[:-1])
    ax_radar.set_xticklabels(lista_metriche, fontweight='bold', fontsize=9)
    ax_radar.tick_params(axis='x', pad=13)
    # Usiamo get_xticklabels() per essere sicuri di non avere stringhe
    for label, angle in zip(ax_radar.get_xticklabels(), angoli):
        angle_deg = np.rad2deg(angle)

        # LOGICA PER "ABBASSARE" O "ALZARE"
        if 90 < angle_deg < 270:
            # set_verticalalignment sposta il testo rispetto al punto di ancoraggio
            label.set_verticalalignment('bottom')
        else:
            label.set_verticalalignment('top')
            label.set_y(label.get_position()[1] - 0.2)

            # Allineamento orizzontale completo
        if angle_deg == 0 or angle_deg == 180:
            label.set_horizontalalignment('center')
        elif 0 < angle_deg < 180:
            label.set_horizontalalignment('center')
        else:
            label.set_horizontalalignment('right')

    ax_radar.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("temp_radar.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Violin Plot
    fig, ax_violin = plt.subplots(figsize=(6, 4))
    dati_violin = [df_raw[df_raw["Metrica"] == m]["Durata (s)"].values for m in lista_metriche]
    parts = ax_violin.violinplot(dati_violin, showmeans=True, showextrema=True)
    for pc in parts['bodies']: pc.set_facecolor(COLOR_SUCCESS); pc.set_edgecolor('none'); pc.set_alpha(0.6)
    parts['cmeans'].set_color(COLOR_FAIL)
    ax_violin.set_title("Time Distribution per Metric", fontsize=20, pad=40, fontweight='bold')
    ax_violin.set_xticks(np.arange(1, len(lista_metriche) + 1))
    ax_violin.set_xticklabels(lista_metriche, rotation=0, ha="center", fontweight='bold')
    ax_violin.set_ylabel("Secondi")
    plt.tight_layout()
    plt.savefig("temp_violin.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Heatmap con altezza dinamica (circa 0.3 pollici per ogni domanda, con un minimo di 5 pollici)
    altezza_dinamica = max(5.0, len(df_master) * 0.3)
    fig, ax_heat = plt.subplots(figsize=(7.5, altezza_dinamica))
    df_heat = df_master.set_index("Input")[lista_metriche]
    labels_y = (df_heat.index.str.slice(0, 50) + "...").tolist()
    dati_matrice = df_heat.values
    im = ax_heat.imshow(dati_matrice, cmap="GnBu", aspect="auto", vmin=0, vmax=1)
    ax_heat.set_xticks(np.arange(len(lista_metriche)))
    ax_heat.set_yticks(np.arange(len(labels_y)))
    ax_heat.set_xticklabels(lista_metriche, fontsize= 7.2, fontweight='bold')
    ax_heat.set_yticklabels(labels_y, fontsize=8)
    plt.setp(ax_heat.get_xticklabels(), rotation=0, ha="center")
    for i in range(len(labels_y)):
        for j in range(len(lista_metriche)):
            valore = dati_matrice[i, j]
            colore_testo = "white" if valore > 0.6 else "black"
            ax_heat.text(j, i, f"{valore:.2f}", ha="center", va="center", color=colore_testo, fontsize=8)
    fig.colorbar(im, ax=ax_heat, label="Punteggio")
    plt.tight_layout()
    plt.savefig("temp_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# FASE 2: MOTORE DI IMPAGINAZIONE (FPDF)
# ==========================================
class PDFReport(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("helvetica", "B", 8)
            self.set_text_color(*NORD_CYAN_RGB)
            self.cell(0, 10, "MIMIR RAG - EVALUATION REPORT", align="R")
            self.ln(10)

    def footer(self):
        if self.page_no() > 1:
            self.set_y(-15)
            self.set_font("helvetica", "I", 8)
            self.set_text_color(*NORD_CYAN_RGB)
            self.cell(0, 10, f"Pagina {self.page_no()}", align="C")


def draw_key_takeaway(pdf, testo):
    y_pos = 250
    pdf.set_y(y_pos)
    pdf.set_fill_color(*NORD_DARK_RGB)
    pdf.rect(15, y_pos, 180, 20, 'F')
    pdf.set_xy(20, y_pos + 3)
    pdf.set_font("helvetica", "B", 10)
    pdf.set_text_color(*NORD_CYAN_RGB)
    pdf.cell(0, 5, "KEY TAKEAWAY", new_x="LMARGIN", new_y="NEXT")
    pdf.set_xy(20, y_pos + 8)
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(*NORD_LIGHT_RGB)
    pdf.multi_cell(170, 5, testo)


def draw_metadata_table(pdf):
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(*NORD_DARK_RGB)
    pdf.cell(0, 8, "Document Control", new_x="LMARGIN", new_y="NEXT")

    metadati = [
        ("Project / Module", "Mimir RAG - Task Specific Module"),
        ("Model Version", "v1.0-RC1"),
        ("Evaluation Framework", "DeepEval (LLM-as-a-Judge)"),
        ("Target Documents", "Corporate Knowledge Base (IT)"),
        ("Date of Report", "Marzo 2026")
    ]

    pdf.set_font("helvetica", "", 10)
    for i, (chiave, valore) in enumerate(metadati):
        if i % 2 == 0:
            pdf.set_fill_color(245, 247, 250)
        else:
            pdf.set_fill_color(255, 255, 255)

        pdf.set_text_color(*NORD_TEXT_RGB)
        pdf.cell(50, 7, f"  {chiave}", border=0, fill=True)
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(130, 7, f"  {valore}", border=0, fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 10)
    pdf.ln(8)


def costruisci_pdf(df_raw, df_master, lista_metriche):
    print("📄 Costruzione del documento PDF...")
    pdf = PDFReport(orientation="P", unit="mm", format="A4")

    # --- COPERTINA ---
    pdf.add_page()
    pdf.set_fill_color(*NORD_DARK_RGB)
    pdf.rect(0, 0, 210, 297, 'F')

    pdf.set_y(100)
    pdf.set_text_color(*NORD_LIGHT_RGB)
    pdf.set_font("helvetica", "B", 38)
    pdf.cell(0, 15, "ENVIRONMENT", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 15, "ANNUAL REPORT", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_y(140)
    pdf.set_text_color(*NORD_CYAN_RGB)
    pdf.set_font("helvetica", "", 18)
    pdf.cell(0, 10, "MIMIR RAG SYSTEM EVALUATION", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_y(260)
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(*NORD_LIGHT_RGB)
    pdf.cell(0, 5, "Internal Use Only", align="C", new_x="LMARGIN", new_y="NEXT")

    # --- PAGINA 2: METADATI & EXECUTIVE SUMMARY ---
    pdf.add_page()
    draw_metadata_table(pdf)

    pdf.set_text_color(*NORD_DARK_RGB)
    pdf.set_font("helvetica", "B", 24)
    pdf.cell(0, 15, "1. Executive Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    x_left = 15
    w_left = 85
    pdf.set_xy(x_left, pdf.get_y())
    pdf.set_font("helvetica", "B", 14)
    pdf.multi_cell(w_left, 10, "Introduction", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(*NORD_TEXT_RGB)
    testo_intro = (
        "Nel corso di questo test di validazione offline, abbiamo sottoposto il modulo task-specific di Mimir a una "
        "serie di query complesse basate sui documenti aziendali. L'obiettivo di questa prima fase e' dimostrare la "
        "capacita' del sistema di recuperare il contesto esatto e di non produrre allucinazioni."
    )
    # L'opzione new_x/new_y qui non è necessaria perché lo riportiamo a x=15 manualmente o c'è ln() dopo.
    pdf.multi_cell(w_left, 6, testo_intro, align="J")

    pdf.ln(8)
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(*NORD_DARK_RGB)
    pdf.cell(w_left, 8, "Our Progress (KPIs):", new_x="LMARGIN", new_y="NEXT")

    totale_domande = len(df_master)
    score_globale = df_raw["Punteggio"].mean() * 100
    tempo_medio = df_master["Tempo_Totale_Domanda_s"].mean()

    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(*NORD_TEXT_RGB)
    pdf.cell(w_left, 6, f"- Total Queries: {totale_domande}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(w_left, 6, f"- Evaluated Metrics: {len(lista_metriche)}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(w_left, 6, f"- Avg. Latency: {tempo_medio:.2f} s", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(5)
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(*NORD_CYAN_RGB)
    pdf.cell(w_left, 10, f"QUALITY SCORE: {score_globale:.1f}/100", new_x="LMARGIN", new_y="NEXT")

    pdf.image("temp_pie.png", x=110, y=80, w=85)
    pdf.image("temp_scores.png", x=110, y=160, w=85)
    draw_key_takeaway(pdf,
                      "Il tasso di successo globale dimostra una baseline solida. L'architettura RAG è pronta per un'ottimizzazione mirata e per scalare su documenti più complessi.")

    # --- PAGINA 3: ANALISI DIMENSIONALE ---
    pdf.add_page()
    pdf.set_text_color(*NORD_DARK_RGB)
    pdf.set_font("helvetica", "B", 24)
    pdf.cell(0, 15, "2. Performance & Metrics Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(*NORD_TEXT_RGB)
    intro_metriche = "Per valutare oggettivamente le prestazioni abbiamo adottato il framework DeepEval, analizzando tre dimensioni fondamentali dell'architettura:"
    pdf.multi_cell(0, 5, intro_metriche, align="J")
    pdf.ln(4)

    def draw_metric_card(title, description, color):
        y_start = pdf.get_y()
        pdf.set_fill_color(*color)
        pdf.rect(15, y_start, 2, 12, 'F')
        pdf.set_xy(20, y_start)
        pdf.set_font("helvetica", "B", 11)
        pdf.set_text_color(*NORD_DARK_RGB)
        pdf.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_xy(20, y_start + 6)
        pdf.set_font("helvetica", "", 9)
        pdf.set_text_color(*NORD_TEXT_RGB)
        pdf.multi_cell(0, 4.5, description, align="J")
        pdf.ln(4)

    draw_metric_card("Faithfulness Metric",
                     "Misura l'accuratezza fattuale. Valuta se l'output dell'LLM è deducibile dal contesto recuperato, penalizzando severamente le allucinazioni.",
                     NORD_GREEN_RGB)
    draw_metric_card("Contextual Relevancy",
                     "Valuta l'efficienza pura del Retriever vettoriale, assicurandosi che il modello riceva solo nodi di testo pertinenti alla query.",
                     NORD_CYAN_RGB)
    draw_metric_card("Answer Relevancy",
                     "Analizza la qualità della risposta finale rispetto alla domanda. Penalizza le risposte evasive, garantendo soluzioni dirette.",
                     NORD_YELLOW_RGB)
    pdf.ln(2)

    y_grafici = pdf.get_y()
    pdf.image("temp_radar.png", x=15, y=y_grafici + 10, w=85)
    pdf.image("temp_violin.png", x=105, y=y_grafici + 10, w=95)

    draw_key_takeaway(pdf,
                      "Mimir eccelle nella fedeltà (non allucina). Il tempo di latenza si mantiene stabile e distribuito equamente senza code anomale critiche.")

    # --- PAGINA 4: HEATMAP ---
    pdf.add_page()
    pdf.set_text_color(*NORD_DARK_RGB)
    pdf.set_font("helvetica", "B", 24)
    pdf.cell(0, 15, "3. Detailed Matrix", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(*NORD_TEXT_RGB)
    testo_matrice = (
        "La matrice sottostante permette un'analisi granulare di ogni singolo test. Le aree scure "
        "indicano alte performance (score vicino a 1.0), fornendo una mappa visiva immediata delle criticità "
        "o dei fallimenti puntuali su specifiche domande."
    )
    pdf.multi_cell(0, 6, testo_matrice, align="J")
    pdf.ln(5)

    pdf.image("temp_heatmap.png", x=15, y=pdf.get_y(), w=180)
    draw_key_takeaway(pdf,
                      "Le zone chiare della Heatmap identificano cluster di vulnerabilità: queste domande specifiche richiederanno un tuning della chunking strategy.")

    # --- PAGINA 5: TOP & FLOP (FIX MARGINI MULTI_CELL) ---
    pdf.add_page()
    pdf.set_text_color(*NORD_DARK_RGB)
    pdf.set_font("helvetica", "B", 24)
    pdf.cell(0, 15, "4. Extreme Case Studies", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    df_sorted = df_raw.sort_values(by="Punteggio", ascending=False)
    top_2 = df_sorted.head(2)
    bottom_2 = df_sorted.tail(2)

    def stampa_blocco_testo_moderno(riga, titolo, colore_rgb):
        pdf.ln(4)

        # Salto pagina intelligente se c'è troppo poco spazio per l'intera card (evita di spezzarla)
        if pdf.get_y() > 220:
            pdf.add_page()

        # Pulizia stringhe (rimuove "a capo" accidentali dall'LLM)
        input_pulito = str(riga['Input']).encode('latin-1', 'replace').decode('latin-1').replace('\n', ' ').strip()
        mot_pulita = str(riga['Motivazione']).encode('latin-1', 'replace').decode('latin-1').replace('\n', ' ').strip()

        # Riportiamo forzatamente il cursore a sinistra per evitare errori FPDF
        pdf.set_x(pdf.l_margin)

        # HEADER DELLA CARD
        pdf.set_fill_color(*colore_rgb)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("helvetica", "B", 10)

        # Usando new_x="LMARGIN" e new_y="NEXT" obblighiamo multi_cell a tornare sempre a sinistra
        header_text = f"   {titolo}   |   Score: {riga['Punteggio']:.2f}   |   {riga['Metrica']}"
        pdf.multi_cell(0, 8, header_text, fill=True, align="L", new_x="LMARGIN", new_y="NEXT")

        # CORPO DELLA CARD (Sfondo Grigio)
        pdf.set_fill_color(245, 247, 250)

        # 1. Query
        pdf.set_font("helvetica", "B", 9)
        pdf.set_text_color(*NORD_DARK_RGB)
        pdf.multi_cell(0, 6, "   Query Utente:", fill=True, align="L", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("helvetica", "", 9)
        pdf.set_text_color(*NORD_TEXT_RGB)
        pdf.multi_cell(0, 5, "   " + input_pulito, fill=True, align="L", new_x="LMARGIN", new_y="NEXT")

        # 2. Justification
        pdf.set_font("helvetica", "B", 9)
        pdf.set_text_color(*NORD_DARK_RGB)
        pdf.multi_cell(0, 6, "   LLM Justification:", fill=True, align="L", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("helvetica", "I", 9)
        pdf.set_text_color(*NORD_TEXT_RGB)
        pdf.multi_cell(0, 5, "   " + mot_pulita, fill=True, align="L", new_x="LMARGIN", new_y="NEXT")

        # Chiusura visiva della card
        pdf.multi_cell(0, 3, "", fill=True, align="L", new_x="LMARGIN", new_y="NEXT")

    for _, row in top_2.iterrows():
        stampa_blocco_testo_moderno(row, "HIGHEST PERFORMANCE", NORD_GREEN_RGB)

    for _, row in bottom_2.iterrows():
        stampa_blocco_testo_moderno(row, "CRITICAL ISSUE", NORD_RED_RGB)

    pdf.output(PDF_OUTPUT)
    print(f"✅ Report Ufficiale salvato come: {PDF_OUTPUT}")


def main():
    try:
        df_raw = pd.read_csv(CSV_FILE, sep=";")
    except FileNotFoundError:
        print(f"⚠️ Errore: File {CSV_FILE} non trovato.")
        return

    df_raw["Metrica"] = df_raw["Metrica"].str.replace("Metric", "")

    df_master, lista_metriche = prepara_dati_scalabili(df_raw)

    genera_immagini_temporanee(df_raw, df_master, lista_metriche)
    costruisci_pdf(df_raw, df_master, lista_metriche)

    for file in ["temp_pie.png", "temp_radar.png", "temp_violin.png", "temp_heatmap.png", "temp_scores.png"]:
        if os.path.exists(file):
            os.remove(file)


if __name__ == "__main__":
    main()