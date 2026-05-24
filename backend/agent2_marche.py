import requests 
 
OLLAMA_URL = "http://localhost:11434/api/generate" 
 
def invoke_llm(prompt): 
    try: 
        response = requests.post( 
            OLLAMA_URL, 
            json={
                "model": "llama3.2", 
                "prompt": prompt, 
                "stream": False,
                "options": {
                    "num_ctx": 4096,
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            }, 
            timeout=600 
        ) 
        if response.status_code == 200: 
            return response.json()["response"] 
        return f"Erreur Ollama (Status: {response.status_code})" 
    except requests.exceptions.Timeout:
        return "Erreur: Le délai d'attente (timeout) a été dépassé après 10 minutes. Le modèle Llama3.2:3b est peut-être trop lent pour votre machine ou la liste d'avis est trop longue."
    except Exception as e: 
        return f"Erreur: {str(e)}" 
 
def generer_rapport_marche(resultats_agent1: str) -> str: 
    prompt = f""" 
 ROLE: Market Analysis Expert
 
 INSTRUCTIONS: 
 Your role is to write a clear and professional report based on the data from Agent 1. 
 
 ABSOLUTE RULE: Totally ignore reviews marked "Neutral". Never include them in your analysis. 
 
 1. PRODUCT STRENGTHS: 
    - Write several strengths in the form of medium-length sentences based on "Positive" reviews. 
    - Reformulate the reviews professionally (no raw copy-pasting like "I love"). 
 
 2. PRODUCT WEAKNESSES: 
    - Write several weaknesses in the form of medium-length sentences based on "Negative" reviews. 
    - Reformulate the technical issues professionally. 
 
 3. MARKET POSITION: 
    - Recopy EXACTLY the percentages provided by Agent 1 (Positive, Negative, Neutral). 
 
 4. IMPROVEMENT RECOMMENDATIONS: 
    - Propose clear recommendations to correct the identified weaknesses. 
 
 5. CUSTOMER SATISFACTION SCORE: 
    - Take the percentage of "Positive" reviews, divide it by 10, and display the result out of 10 (e.g., 45% = 4.5/10). 
 
 Generate ONLY the report in the requested format, without adding any text before or after. 
 
 FORMAT: 
 
 MARKET ANALYSIS REPORT 
 
 PRODUCT STRENGTHS: 
 - [Strength 1] 
 - [Strength 2]... 
 
 PRODUCT WEAKNESSES: 
 - [Weakness 1] 
 - [Weakness 2]... 
 
 MARKET POSITION: 
 - Positive: [X]% 
 - Negative: [X]% 
 - Neutral: [X]% 
 
 IMPROVEMENT RECOMMENDATIONS: 
 - [Recommendation 1] 
 - [Recommendation 2]... 
 
 CUSTOMER SATISFACTION SCORE: [Score]/10 
 
 DATA FROM AGENT 1: 
 {resultats_agent1} 
 """ 
    return invoke_llm(prompt) 
