import torch 
from transformers import BertTokenizer, BertForSequenceClassification 
import torch.nn.functional as F 
import os 
 
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'modele', 'bert_model_complet') 
DEVICE     = torch.device("cpu") 
LABELS     = {0: "Negatif", 1: "Neutre", 2: "Positif"} 
 
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH) 
model     = BertForSequenceClassification.from_pretrained(MODEL_PATH) 
model.to(DEVICE) 
model.eval() 
print("Modele BERT charge depuis bert_model_complet !") 
 
def predict(texte: str) -> dict: 
    if not texte or len(texte.strip()) < 3: 
        return {"label": "Neutre", "score": 0.0, "label_id": 1} 
    
    encoding = tokenizer( 
        texte, 
        truncation=True, 
        padding="max_length", 
        max_length=128, 
        return_tensors="pt" 
    ) 
    with torch.no_grad(): 
        outputs = model( 
            input_ids=encoding["input_ids"].to(DEVICE), 
            attention_mask=encoding["attention_mask"].to(DEVICE) 
        ) 
    probs    = F.softmax(outputs.logits, dim=1) 
    label_id = torch.argmax(probs, dim=1).item() 
    score    = round(probs[0][label_id].item() * 100, 2) 
    
    return { 
        "label"    : LABELS[label_id], 
        "score"    : score, 
        "label_id" : label_id, 
        "all_scores": { 
            "Negatif": round(probs[0][0].item() * 100, 2), 
            "Neutre" : round(probs[0][1].item() * 100, 2), 
            "Positif": round(probs[0][2].item() * 100, 2) 
        } 
    } 
 
if __name__ == "__main__": 
    tests = [ 
        "This product is excellent, I love it!", 
        "Very disappointed, waste of money", 
        "It is okay, nothing special" 
    ] 
    for t in tests: 
        r = predict(t) 
        print(f"{t[:40]} -> {r['label']} ({r['score']}%)") 
