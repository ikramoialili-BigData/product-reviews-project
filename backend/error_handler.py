import os
from .logger import log_action

def validate_text(text):
    """
    Vérifie et nettoie le texte d'entrée.
    - Vérifie si vide
    - Vérifie si trop court (< 3 mots)
    - Tronque si trop long (> 500 mots)
    """
    if not text or not isinstance(text, str):
        error_msg = "Erreur: Le texte est vide ou invalide."
        log_action("ErrorHandler", "validate_text", text, error_msg, "error")
        return None, error_msg
    
    words = text.split()
    if len(words) < 3:
        error_msg = "Erreur: Le texte est trop court (minimum 3 mots)."
        log_action("ErrorHandler", "validate_text", text, error_msg, "error")
        return None, error_msg
    
    if len(words) > 500:
        text = " ".join(words[:500])
        # On ne retourne pas d'erreur ici, juste le texte tronqué
        
    return text, None

def handle_api_error(error, agent_name):
    """Gère les erreurs liées aux APIs."""
    error_msg = f"Erreur API ({agent_name}): {str(error)}"
    log_action(agent_name, "api_call", "N/A", error_msg, "error")
    return error_msg

def handle_model_error(error, model_name="BERT"):
    """Gère les erreurs de chargement ou d'inférence du modèle."""
    error_msg = f"Erreur Modèle ({model_name}): {str(error)}"
    log_action("System", "model_load/inference", model_name, error_msg, "error")
    return error_msg

def safe_execution(func, *args, **kwargs):
    """Exécute une fonction en capturant les exceptions pour éviter le crash."""
    try:
        return func(*args, **kwargs), None
    except Exception as e:
        return None, str(e)
