import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotx
from fpdf import FPDF

NORD_THEME = {
    'DARK': (46, 52, 64),
    'LIGHT': (236, 239, 244),
    'CYAN': (136, 192, 208),
    'TEXT': (76, 86, 106),
    'SUCCESS': '#A3BE8C',
    'FAIL': '#BF616A',
    'SUCCESS_RGB': (163, 190, 140),
    'FAIL_RGB': (191, 97, 106)
}


class _MimirCorePDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(15, 15, 15)
        self.set_auto_page_break(False)
        self.alias_nb_pages()

    def header(self):
        if self.page_no() > 1:
            self.set_font("helvetica", "B", 8)
            self.set_text_color(*NORD_THEME['TEXT'])
            self.cell(0, 5, "MIMIR EVALUATION REPORT | REPORT AUTOMATICO", align="R")
            self.line(15, 22, 195, 22)
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "B", 8)
        self.set_text_color(*NORD_THEME['TEXT'])
        self.cell(0, 10, f"Pagina {self.page_no()} di {{nb}}", align="C")


class MimirPDFPrinter:
    def __init__(self, csv_file_path: str, output_pdf_path: str):
        # Attributi protetti
        self._csv_file = csv_file_path
        self._output_pdf = os.path.abspath(output_pdf_path)
        self._output_dir = os.path.dirname(self._output_pdf)
        self._report_name = os.path.splitext(os.path.basename(self._output_pdf))[0]

        os.makedirs(self._output_dir, exist_ok=True)

    def _clean_str(self, testo):
        if pd.isna(testo):
            return ""
        return str(testo).encode('latin-1', 'replace').decode('latin-1').replace('\n', ' ').strip()

    def _prepara_dati(self, df):
        df["Metrica"] = df["Metrica"].str.replace("Metric", "")
        df_punteggi = df.pivot_table(index="Input", columns="Metrica", values="Punteggio",
                                     aggfunc='first').reset_index()
        df_tempi = df.groupby("Input")["Durata (s)"].sum().reset_index()
        df_tempi.rename(columns={"Durata (s)": "Tempo_Totale"}, inplace=True)

        master_df = pd.merge(df_punteggi, df_tempi, on="Input")
        cols_score = [c for c in master_df.columns if c not in ["Input", "Tempo_Totale"]]
        master_df['Score_Medio'] = master_df[cols_score].mean(axis=1)

        return master_df, cols_score

    def _genera_grafici_matematici(self, df_raw, df_master, metriche):
        pie_name = f"{self._report_name}_pie.png"
        bar_name = f"{self._report_name}_metrics_bar.png"
        heat_name = f"{self._report_name}_heatmap.png"

        plt.style.use(matplotx.styles.nord)
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'patch.facecolor': 'white',
            'savefig.facecolor': 'white',
            'text.color': '#2E3440',
            'axes.labelcolor': '#2E3440',
            'xtick.color': '#2E3440',
            'ytick.color': '#2E3440'
        })

        passati = len(df_raw[df_raw["Esito"] == "PASSED"])
        falliti = len(df_raw) - passati

        # --- PIE CHART ---
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.pie([passati, falliti], labels=['PASSED', 'FAILED'], colors=[NORD_THEME['SUCCESS'], NORD_THEME['FAIL']],
               autopct='%1.1f%%', startangle=90, textprops={'weight': 'bold', 'fontsize': 10})
        ax.set_title("Global Success Rate", fontweight='bold', fontsize=12, pad=15)
        fig.tight_layout()
        plt.savefig(os.path.join(self._output_dir, pie_name), dpi=300)
        plt.close()

        # --- BAR CHART ---
        fig, ax = plt.subplots(figsize=(6, 5))
        medie_metriche = df_raw.groupby("Metrica")["Punteggio"].mean()
        bars = ax.bar(medie_metriche.index, medie_metriche.values, color='#88C0D0', width=0.5)
        ax.set_ylim(0, 1.1)
        ax.set_title("Average Score per Metric", fontweight='bold', fontsize=14, pad=15)
        ax.set_ylabel("Punteggio Medio", fontweight='bold', fontsize=10, labelpad=10)
        plt.xticks(rotation=15, ha='right')
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        plt.savefig(os.path.join(self._output_dir, bar_name), dpi=300)
        plt.close()

        # --- HEATMAP ---
        fig, ax = plt.subplots(figsize=(6, 5))
        dati_matrice = df_master.set_index("Input")[metriche].head(30)
        im = ax.imshow(dati_matrice.values, cmap="GnBu", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(metriche)))
        ax.set_xticklabels(metriche, fontsize=9, fontweight='bold')
        ax.set_yticks(np.arange(len(dati_matrice)))
        ax.set_yticklabels([f"Q{i + 1}" for i in range(len(dati_matrice))], fontsize=9)
        ax.set_title("Performance Heatmap", fontweight='bold', fontsize=12, pad=15)
        ax.set_ylabel("Trace", fontweight='bold', fontsize=10, labelpad=10)
        plt.xticks(rotation=15, ha='right')
        for i in range(len(dati_matrice)):
            for j in range(len(metriche)):
                val = dati_matrice.values[i, j]
                color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=6, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8, label="Score (0.0 - 1.0)")
        fig.tight_layout()
        plt.savefig(os.path.join(self._output_dir, heat_name), dpi=300)
        plt.close()

    def genera_report(self):
        """Metodo pubblico esposto per la generazione del file PDF"""
        if not os.path.exists(self._csv_file):
            raise FileNotFoundError(f"File sorgente CSV non trovato: {self._csv_file}")

        df_raw = pd.read_csv(self._csv_file, sep=";")
        df_master, metriche = self._prepara_dati(df_raw)

        self._genera_grafici_matematici(df_raw, df_master, metriche)

        pdf = _MimirCorePDF()
        timestamp_ora = datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")

        pie_name = f"{self._report_name}_pie.png"
        bar_name = f"{self._report_name}_metrics_bar.png"
        heat_name = f"{self._report_name}_heatmap.png"

        # PAGINA 1
        pdf.add_page()
        pdf.set_fill_color(*NORD_THEME['DARK'])
        pdf.rect(0, 0, 210, 38, 'F')

        pdf.set_xy(15, 10)
        pdf.set_font("helvetica", "B", 20)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 8, "MIMIR EVALUATION REPORT", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "B", 10)
        pdf.set_text_color(*NORD_THEME['CYAN'])
        pdf.cell(0, 5, f"DATA DI GENERAZIONE: {timestamp_ora}")

        pdf.set_xy(15, 48)
        pdf.set_font("helvetica", "B", 11)
        pdf.set_text_color(*NORD_THEME['DARK'])
        pdf.cell(0, 6, "DOCUMENT CONTROL & CONFIGURATION", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        insieme_metriche = ", ".join(metriche)

        metadati = [
            ("Framework di Valutazione", "DeepEval"),
            ("File Dati Sorgente", os.path.basename(self._csv_file)),
            ("Stato del Testset", f"Esecuzione completata con {len(df_master)} tracce analizzate"),
            ("Metriche Valutate", insieme_metriche)
        ]

        for idx, (chiave, valore) in enumerate(metadati):
            bg = (245, 247, 250) if idx % 2 == 0 else (255, 255, 255)
            pdf.set_fill_color(*bg)
            pdf.set_font("helvetica", "", 9)
            pdf.set_text_color(*NORD_THEME['TEXT'])
            pdf.cell(45, 6, f"  {chiave}", fill=True)
            pdf.set_font("helvetica", "B", 9)
            pdf.set_text_color(*NORD_THEME['DARK'])
            pdf.cell(135, 6, f"  {valore}", fill=True, new_x="LMARGIN", new_y="NEXT")

        pdf.set_xy(15, 88)
        pdf.set_font("helvetica", "B", 14)
        pdf.set_text_color(*NORD_THEME['DARK'])
        pdf.cell(0, 8, "1. Global evaluation results analysis", new_x="LMARGIN", new_y="NEXT")
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(6)

        y_col = pdf.get_y()
        pdf.image(os.path.join(self._output_dir, pie_name), x=18, y=y_col, w=65)

        pdf.set_xy(92, y_col + 5)
        pdf.set_font("helvetica", "B", 11)
        pdf.set_text_color(*NORD_THEME['DARK'])
        pdf.cell(0, 6, "Analisi dell'Affidabilita' Globale", new_x="LMARGIN", new_y="NEXT")

        pdf.set_xy(92, pdf.get_y())
        pdf.set_font("helvetica", "", 10)
        pdf.set_text_color(*NORD_THEME['TEXT'])
        success_pct = (len(df_raw[df_raw["Esito"] == "PASSED"]) / len(df_raw)) * 100
        desc_pie = (
            f"Il grafico evidenzia il tasso di superamento dei test basato sulle soglie configurate "
            f"nella libreria di valutazione. Attualmente, la compliance del sistema si attesta sul valore del "
            f"{success_pct:.1f}% di risposte conformi ai requisiti di qualità. I casi non conformi (FAILED) indicano "
            f"allucinazioni semantiche o risposte evasive che richiedono interventi di ottimizzazione della knowledge base."
        )
        pdf.multi_cell(103, 5, desc_pie, align="J")

        pdf.set_xy(15, 165)
        pdf.set_fill_color(*NORD_THEME['LIGHT'])
        pdf.rect(92, 150, 103, 14, 'F')
        pdf.set_xy(76, 152)
        pdf.set_font("helvetica", "B", 9)
        pdf.set_text_color(*NORD_THEME['TEXT'])
        pdf.cell(60, 4, "SCORE MEDIO", align="C")
        pdf.cell(60, 4, "LATENZA MEDIA DI VALUTAZIONE", align="C")

        pdf.set_x(76)
        pdf.set_font("helvetica", "B", 12)
        pdf.set_text_color(*NORD_THEME['DARK'])
        pdf.cell(60, 15, f"{df_raw['Punteggio'].mean():.2f} / 1.00", align="C")
        pdf.cell(60, 15, f"{df_master['Tempo_Totale'].mean():.2f} s", align="C")

        # PAGINA 2
        pdf.add_page()
        pdf.set_xy(15, 30)
        pdf.set_font("helvetica", "B", 14)
        pdf.set_text_color(*NORD_THEME['DARK'])
        pdf.cell(0, 8, "2. Performance Metrics Analysis", new_x="LMARGIN", new_y="NEXT")
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(6)

        y_col2 = pdf.get_y()
        pdf.image(os.path.join(self._output_dir, bar_name), x=18, y=y_col2, w=85)

        pdf.set_xy(105, y_col2 + 5)
        pdf.set_font("helvetica", "B", 11)
        pdf.cell(0, 6, "Interpretazione dell'Istogramma", new_x="LMARGIN", new_y="NEXT")

        pdf.set_xy(105, pdf.get_y())
        pdf.set_font("helvetica", "", 9)
        pdf.set_text_color(*NORD_THEME['TEXT'])
        desc_bar = (
            "L'istogramma scompone analiticamente le performance matematiche medie per ciascuna "
            "delle metriche configurate. Questo permette di isolare se le criticità prestazionali dell'agente "
            "risiedano nella fase di recupero delle informazioni dal database vettoriale o nella fase "
            "di generazione e sintesi linguistica."
        )
        pdf.multi_cell(90, 4.5, desc_bar, align="J")

        pdf.set_xy(15, 120)
        pdf.set_font("helvetica", "B", 14)
        pdf.set_text_color(*NORD_THEME['DARK'])
        pdf.cell(0, 6, "3. Performance Heatmap", new_x="LMARGIN", new_y="NEXT")
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(4)

        y_heat = pdf.get_y()
        pdf.image(os.path.join(self._output_dir, heat_name), x=15, y=y_heat, w=115)

        pdf.set_xy(135, y_heat + 10)
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(0, 5, "Interpretazione della Mappa", new_x="LMARGIN", new_y="NEXT")
        pdf.set_xy(135, pdf.get_y())
        pdf.set_font("helvetica", "", 9)
        pdf.set_text_color(*NORD_THEME['TEXT'])
        desc_heat = (
            "La matrice traccia l'andamento di ogni singola domanda (asse Y) rispetto agli "
            "score ottenuti (asse X). "
            "I cluster con colorazione piu' chiara identificano cali di performance semantica che richiedono "
            "interventi correttivi prioritari."
        )
        pdf.multi_cell(60, 4.5, desc_heat, align="J")

        # PAGINA 3
        pdf.add_page()
        pdf.set_xy(15, 30)
        pdf.set_font("helvetica", "B", 14)
        pdf.set_text_color(*NORD_THEME['DARK'])
        pdf.cell(0, 8, "4. Evaluation Log by Query", new_x="LMARGIN", new_y="NEXT")
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(6)

        def render_query_card(input_text, gruppo_metriche):
            spazio_minimo_richiesto = 40
            if pdf.get_y() + spazio_minimo_richiesto > 270:
                pdf.add_page()
                pdf.set_y(30)

            score_medio = gruppo_metriche['Punteggio'].mean()

            pdf.set_fill_color(*NORD_THEME['DARK'])
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("helvetica", "B", 10)
            query_troncata = (input_text[:50] + '...') if len(input_text) > 80 else input_text
            pdf.cell(0, 7, self._clean_str(f"  QUERY: {query_troncata} | Score Medio: {score_medio:.2f}"), fill=True,
                     new_x="LMARGIN", new_y="NEXT")

            for _, riga in gruppo_metriche.iterrows():
                if pdf.get_y() > 270:
                    pdf.add_page()
                    pdf.set_y(30)

                is_success = riga['Punteggio'] >= riga.get('Soglia', 0.7)
                color_row = NORD_THEME['SUCCESS_RGB'] if is_success else NORD_THEME['FAIL_RGB']
                colonna_text = "Ragionamento" if "Ragionamento" in riga else "Motivazione"

                pdf.set_fill_color(*color_row)
                pdf.set_text_color(255, 255, 255)
                pdf.set_font("helvetica", "B", 9)
                pdf.cell(0, 6, f"    > Metrica: {riga['Metrica']} (Score: {riga['Punteggio']:.2f})", fill=True,
                         new_x="LMARGIN", new_y="NEXT")

                pdf.set_fill_color(*NORD_THEME['LIGHT'])
                pdf.set_text_color(*NORD_THEME['TEXT'])
                pdf.set_font("helvetica", "I", 9)
                pdf.multi_cell(0, 4.5, "    " + self._clean_str(riga[colonna_text]), fill=True, align="L",
                               new_x="LMARGIN", new_y="NEXT")
                pdf.ln(1)

            pdf.ln(4)

        for input_text, gruppo in df_raw.groupby("Input"):
            render_query_card(input_text, gruppo)

        pdf.ln(5)
        pdf.set_font("helvetica", "B", 10)
        pdf.set_text_color(*NORD_THEME['DARK'])
        pdf.cell(0, 5, "Fine del Report di Valutazione", align="C", new_x="LMARGIN", new_y="NEXT")

        pdf.output(self._output_pdf)