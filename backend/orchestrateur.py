import os 
from backend.logger import log_action 
from backend.modele_bert import predict 
from backend.agent1_sentiment import analyser_avis
from backend.agent2_marche import generer_rapport_marche
from fpdf import FPDF 
from datetime import datetime 
 
def generer_rapport_final(resultats_agent1: str, resultats_agent2: str) -> str: 
    from datetime import datetime 
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
    
    rapport = f""" 
 ╔══════════════════════════════════════════════════════╗ 
 ║         PRODUCT REVIEW INTELLIGENCE REPORT           ║ 
 ║                  {timestamp}                
 ╚══════════════════════════════════════════════════════╝ 
 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 1. SENTIMENT ANALYSIS (Agent 1 - BERT) 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 {resultats_agent1} 
 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 2. MARKET ANALYSIS (Agent 2 - Llama3.2:1b) 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 {resultats_agent2} 
 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 3. SYSTEM INFO 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 - BERT Model  : Fine-tuned bert-base-uncased (87.78% accuracy) 
 - LLM Backend : Ollama llama3.2:1b 
 - Framework   : Multi-Agent System 
 - Generated   : {timestamp} 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 """ 
    
        # Save as PDF 
    try: 
        import os 
        from fpdf import FPDF 
        os.makedirs("rapports", exist_ok=True) 
        pdf = FPDF() 
        pdf.add_page() 
        pdf.set_font("Arial", size=11) 
        for line in rapport.split('\n'): 
            pdf.cell(200, 8, txt=line.encode('latin-1', 'replace').decode('latin-1'), ln=True) 
        pdf_path = f"rapports/rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf" 
        pdf.output(pdf_path) 
        print(f"PDF saved: {pdf_path}") 
    except Exception as e: 
        print(f"PDF error: {e}") 
    
    return rapport 
 
class Orchestrateur: 
    def __init__(self): 
        self.model = "llama3.2" 
     
    def lancer_analyse(self, liste_avis): 
        try: 
            log_action( 
                "Orchestrateur", 
                "start", 
                f"{len(liste_avis)} reviews", 
                "In progress" 
            ) 
            
            resultats_bert = [] 
            for avis in liste_avis: 
                if avis.strip(): 
                    try: 
                        result = predict(avis) 
                        # Translate BERT label to English
                        bert_label = result["label"]
                        if bert_label == "Positif": label = "Positive"
                        elif bert_label == "Negatif": label = "Negative"
                        else: label = "Neutral"
                        confidence = result["score"]
                        resultats_bert.append(f"- {avis[:60]} -> {label} ({confidence}%)")
                    except Exception as e: 
                        resultats_bert.append(f"- {avis[:60]} -> Neutral") 
            
            bert_text = "\n".join(resultats_bert) 
            
            log_action( 
                "Agent1_Sentiment", 
                "bert_analyse", 
                str(len(liste_avis)), 
                bert_text 
            ) 
            
            # Using the new agent for sentiment analysis
            res1 = analyser_avis(liste_avis)
            
            if not res1 or res1 == "None": 
                res1 = "Ollama Error - Check that Ollama is running" 
            
            log_action( 
                "Agent1_Sentiment", 
                "end", 
                bert_text[:100], 
                str(res1)[:100] 
            ) 
            
            # Utilisation du nouvel agent pour l'analyse de marché
            res2 = generer_rapport_marche(res1)
            
            if not res2 or res2 == "None": 
                res2 = "Ollama Error - Check that Ollama is running" 
            
            log_action( 
                "Agent2_Marche", 
                "fin", 
                str(res1)[:100], 
                str(res2)[:100] 
            ) 

            log_action( 
                "Orchestrateur", 
                "fin", 
                "N/A", 
                "Succes" 
            ) 
            
            return res1, res2, "Analyse terminée. Cliquez sur Valider pour voir le rapport." 
             
        except Exception as e: 
            error = str(e) 
            print(f"ERREUR ORCHESTRATEUR: {error}") # Afficher dans le terminal
            log_action( 
                "Orchestrateur", 
                "erreur", 
                "N/A", 
                error, 
                "error" 
            ) 
            return f"Erreur: {error}", f"Erreur: {error}", error 
     
    def generer_rapport_pdf(self, res1, res2): 
        try: 
            pdf = FPDF() 
            pdf.add_page() 
            pdf.set_font("Arial", "B", 16) 
            pdf.cell( 
                200, 10, 
                "Product Review Intelligence", 
                ln=True, 
                align="C" 
            ) 
            pdf.ln(10) 
            pdf.set_font("Arial", "B", 13) 
            pdf.cell( 
                200, 10, 
                "Agent 1 - Sentiment Analysis", 
                ln=True 
            ) 
            pdf.set_font("Arial", size=11) 
            t1 = str(res1).encode( 
                "latin-1","replace" 
            ).decode("latin-1") 
            pdf.multi_cell(0, 8, t1) 
            pdf.ln(5) 
            pdf.set_font("Arial", "B", 13) 
            pdf.cell( 
                200, 10, 
                "Agent 2 - Market Analysis", 
                ln=True 
            ) 
            pdf.set_font("Arial", size=11) 
            t2 = str(res2).encode( 
                "latin-1","replace" 
            ).decode("latin-1") 
            pdf.multi_cell(0, 8, t2) 
            os.makedirs("rapports", exist_ok=True) 
            nom = ( 
                f"rapports/rapport_" 
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}" 
                f".pdf" 
            ) 
            pdf.output(nom) 
            return nom, None 
        except Exception as e: 
            return None, str(e) 
 
def format_final_report(results): 
    return f""" 
{'='*40} 
FINAL REPORT 
{'='*40} 
{results} 
{'='*40} 
""" 
