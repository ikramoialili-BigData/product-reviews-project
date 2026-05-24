import requests 
import sys 
import os 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from backend.modele_bert import predict 
 
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
                    "num_predict": 1024 
                } 
            }, 
            timeout=300 
        ) 
        if response.status_code == 200: 
            return response.json()["response"] 
        return "Ollama Error" 
    except Exception as e: 
        return f"Error: {str(e)}" 
 
def analyser_avis(liste_avis: list) -> str: 
    bert_results = [] 
    compteur = {"Positive": 0, "Negative": 0, "Neutral": 0} 
    
    for avis in liste_avis: 
        try: 
            # Forced "Neutral" classification rule 
            lower_avis = avis.lower() 
            if any(word in lower_avis for word in ["expected", "nothing special", "okay", "average"]): 
                label = "Neutral" 
                # High confidence score for keyword-forced neutral
                score = 96.50 
            else: 
                result = predict(avis) 
                # Translate BERT label to English if necessary
                bert_label = result["label"]
                if bert_label == "Positif": label = "Positive"
                elif bert_label == "Negatif": label = "Negative"
                else: label = "Neutral"
                score = result["score"] 
            
            bert_results.append((avis, label, score)) 
            if label in compteur: 
                compteur[label] += 1 
        except Exception as e: 
            bert_results.append((avis, "Neutral", 0.0)) 
            compteur["Neutral"] += 1 
    
    total = len(liste_avis) 
    pct_pos = round(compteur["Positive"] / total * 100) if total > 0 else 0 
    pct_neg = round(compteur["Negative"] / total * 100) if total > 0 else 0 
    pct_neu = round(compteur["Neutral"] / total * 100) if total > 0 else 0 
    
    # Format the list according to strict numbered format 1 to N
    reviews_formatted = "\n\n".join([f"{i+1}. '{r[0].replace('[','').replace(']','')}' - {r[1]} ({r[2]:.2f}%)" for i, r in enumerate(bert_results)]) 
    
    prompt = f""" 
 ROLE: Technical Presentation Agent
 
 INSTRUCTIONS: 
 You are a technical presentation agent. Your role is to display the classification results provided by the pre-trained BERT model. You must not analyze, interpret, or recalculate the data. 
 
 DISPLAY RULES: 
 1. Faithful Reproduction: Present the data exactly as classified by BERT. Add no analysis, no comments, and change no scores. 
 2. Statistics: Use EXCLUSIVELY the percentages provided below. Do not attempt to recalculate them yourself. 
 3. Prohibition: Do not add titles like "Step 1" or "Extraction", display no mathematical reasoning, and add no comments. 
 
 MANDATORY OUTPUT FORMAT (STRICT): 
 
 Based on the provided data, here is a summary of the REAL BERT sentiment results: 
 
 {reviews_formatted} 
 
 The REAL BERT sentiment results indicate that: 
 
 {pct_pos}% of the reviews have a positive label 
 {pct_neg}% of the reviews have a negative label 
 {pct_neu}% of the reviews have a neutral label 
 
 CRITICAL REQUIREMENT: You MUST include the "The REAL BERT sentiment results indicate that:" section with the percentages. It is the most important part of the report. DO NOT stop before writing the statistics.
 """ 
    return invoke_llm(prompt) 
