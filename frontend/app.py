import gradio as gr 
import sys 
import os 
from datetime import datetime 
 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from backend.orchestrateur import Orchestrateur 
 
orchestrateur = Orchestrateur() 
res1_global = "" 
res2_global = "" 
 
def analyser(avis_text): 
    global res1_global, res2_global 
    if not avis_text.strip(): 
        return "Please enter reviews!", "", gr.update(visible=False), gr.update(visible=False) 
    liste = [a.strip() for a in avis_text.split("\n") if a.strip()] 
    try: 
        res1, res2, _ = orchestrateur.lancer_analyse(liste) 
        res1_global = str(res1) 
        res2_global = str(res2) 
        return res1_global, res2_global, gr.update(visible=True), gr.update(visible=False) 
    except Exception as e: 
        return f"Error: {e}", "", gr.update(visible=False), gr.update(visible=False) 
 
def valider(): 
    global res1_global, res2_global 
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
    
    rapport = f""" 
 ╔══════════════════════════════════════════════════════════════╗ 
 ║          PRODUCT REVIEW INTELLIGENCE - FINAL REPORT          ║ 
 ║                      Generated: {t}              
 ╚══════════════════════════════════════════════════════════════╝ 
 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
  SECTION 1 : SENTIMENT ANALYSIS RESULTS (Agent 1 - BERT) 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 {res1_global} 
 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
  SECTION 2 : MARKET ANALYSIS RESULTS (Agent 2 - Llama3.2) 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 {res2_global} 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 """ 
 
    # Save PDF in separate thread
    import threading 
    def save_pdf(): 
        try: 
            from fpdf import FPDF 
            os.makedirs("rapports", exist_ok=True) 
            pdf = FPDF() 
            pdf.add_page() 
            pdf.set_font("Courier", size=9) 
            clean = rapport.replace('═','=').replace('║','|').replace('━','-').replace('╔','+').replace('╚','+').replace('╗','+').replace('╝','+') 
            for line in clean.split('\n'): 
                pdf.cell(200, 6, txt=line.encode('latin-1','replace').decode('latin-1'), ln=True) 
            pdf.output(f"rapports/rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf") 
            print("PDF saved!") 
        except Exception as e: 
            print(f"PDF error: {e}") 
    threading.Thread(target=save_pdf, daemon=True).start() 
 
    return rapport, gr.update(visible=True), gr.update(visible=False), gr.update(visible=True) 
 
def annuler(): 
    return "", gr.update(visible=False), gr.update(visible=False) 
 
with gr.Blocks(title="Product Review Intelligence") as app: 
    gr.Markdown(""" 
    # 🤖 Product Review Intelligence 
    <p style='text-align: center; color: #3b82f6; font-size: 1.2em;'> 
    Multi-Agent AI System — BERT + Ollama llama3.2 
    </p> 
    """) 
 
    with gr.Group(): 
        gr.Markdown("## 📝 Customer Reviews Input") 
        avis_input = gr.Textbox( 
            label="Customer Reviews", 
            placeholder="Enter one review per line...", 
            lines=8 
        ) 
        analyser_btn = gr.Button("🚀 Analyze Reviews", variant="primary") 
 
    with gr.Group(): 
        gr.Markdown("## 🤖 Agent 1 - Sentiment Analysis (BERT)") 
        agent1_out = gr.Textbox(label="Sentiment Results", lines=8, interactive=False) 
 
    with gr.Group(): 
        gr.Markdown("## 🔍 Agent 2 - Market Analysis (Llama3.2)") 
        agent2_out = gr.Textbox(label="Market Results", lines=12, interactive=False, elem_id="agent2-results") 
 
    with gr.Group(visible=False) as hitl_zone: 
        gr.Markdown("## ⚠️ Human Validation Required (HITL)") 
        gr.Markdown("Agents have completed their analysis. Do you want to generate the final report?") 
        with gr.Row(): 
            valider_btn = gr.Button("✅ Validate and Generate Report", variant="primary") 
            annuler_btn = gr.Button("❌ Cancel", variant="stop") 
 
    with gr.Group(visible=False) as rapport_zone: 
        gr.Markdown("## 📄 Final Report") 
        rapport_out = gr.Textbox(label="Report", lines=20, interactive=False) 
 
    analyser_btn.click( 
        fn=analyser, 
        inputs=[avis_input], 
        outputs=[agent1_out, agent2_out, hitl_zone, rapport_zone] 
    ) 
    valider_btn.click( 
        fn=valider, 
        inputs=[], 
        outputs=[rapport_out, rapport_zone, hitl_zone, rapport_zone] 
    ) 
    annuler_btn.click( 
        fn=annuler, 
        inputs=[], 
        outputs=[rapport_out, rapport_zone, hitl_zone] 
    ) 
 
if __name__ == "__main__": 
    app.launch( 
        server_name="0.0.0.0", 
        server_port=7860, 
        theme=gr.themes.Soft( 
            primary_hue="blue", 
            secondary_hue="indigo", 
            neutral_hue="slate" 
        ), 
        css=""" 
        * { 
            font-family: 'Segoe UI', sans-serif !important; 
        } 
        .gradio-container { 
            max-width: 100% !important; 
            width: 100% !important; 
            margin: auto !important; 
            padding: 20px !important; 
        } 
        .gr-button-primary { 
            background: linear-gradient(90deg, #1e3a8a, #3b82f6) !important; 
            border: none !important; 
            color: white !important; 
            font-weight: bold !important; 
            width: 100% !important; 
        } 
        .gr-button-stop { 
            background: linear-gradient(90deg, #dc2626, #ef4444) !important; 
            border: none !important; 
            color: white !important; 
        } 
        h1 { 
            text-align: center !important; 
            color: #1e3a8a !important; 
            font-size: 2.5em !important; 
            font-weight: bold !important; 
        } 
        h3 { 
            color: #3b82f6 !important; 
        } 
        #agent2-results textarea { 
            overflow-y: auto !important; 
        } 
        footer { 
            display: none !important; 
        } 
        """ 
    ) 
