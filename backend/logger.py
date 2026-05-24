import json
import os
from datetime import datetime

# Chemin vers le dossier des logs
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "actions.json")

def setup_logger():
    """Crée le dossier logs si inexistant."""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

def log_action(agent_name, action, input_data, output_data, status="success"):
    """
    Sauvegarde une action dans le fichier JSON.
    Chaque entrée est sur une nouvelle ligne.
    """
    setup_logger()
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent_name,
        "action": action,
        "input": str(input_data),
        "output": str(output_data),
        "status": status
    }
    
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Erreur lors de l'écriture du log: {e}")

def get_all_logs():
    """Retourne tous les logs sous forme de liste de dictionnaires."""
    logs = []
    if not os.path.exists(LOG_FILE):
        return logs
        
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))
    except Exception as e:
        print(f"Erreur lors de la lecture des logs: {e}")
    
    return logs
