import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

# Configuration
DATASET_PATH = "7817_1.csv"
MODEL_SAVE_PATH = "model.pth"
BERT_MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

class AmazonReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]
        
        encoding = self.tokenizer(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

def prepare_data():
    print("--- Chargement du dataset ---")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset non trouvé à {DATASET_PATH}")
        
    df = pd.read_csv(DATASET_PATH)
    
    # Garder seulement les colonnes utiles
    df = df[['reviews.text', 'reviews.rating']]
    
    # Supprimer les lignes vides
    df = df.dropna()
    
    # Conversion des notes en labels
    # Note 1,2 -> 0 (Négatif)
    # Note 3   -> 1 (Neutre)
    # Note 4,5 -> 2 (Positif)
    def map_rating(rating):
        if rating <= 2: return 0
        if rating == 3: return 1
        return 2
        
    df['label'] = df['reviews.rating'].apply(map_rating)
    
    return train_test_split(df, test_size=0.2, random_state=42)

def train_model():
    # 1. Préparation des données
    df_train, df_test = prepare_data()
    print(f"Train size: {len(df_train)}, Test size: {len(df_test)}")
    
    # 2. Tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    
    # 3. DataLoaders
    train_dataset = AmazonReviewDataset(
        reviews=df_train['reviews.text'].to_numpy(),
        targets=df_train['label'].to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    test_dataset = AmazonReviewDataset(
        reviews=df_test['reviews.text'].to_numpy(),
        targets=df_test['label'].to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 4. Modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=3)
    model = model.to(device)
    
    # 5. Optimiseur
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 6. Boucle d'entraînement
    print(f"--- Début de l'entraînement sur {device} ---")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        print(f"Loss moyenne Epoch {epoch+1}: {total_loss/len(train_loader):.4f}")
        
    # 7. Évaluation
    print("--- Évaluation ---")
    model.eval()
    predictions = []
    real_values = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            real_values.extend(targets.cpu().tolist())
            
    # 8. Rapport et Confusion Matrix
    print("\nClassification Report:")
    print(classification_report(real_values, predictions, target_names=['Négatif', 'Neutre', 'Positif']))
    
    cm = confusion_matrix(real_values, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Négatif', 'Neutre', 'Positif'], yticklabels=['Négatif', 'Neutre', 'Positif'])
    plt.xlabel('Prédiction')
    plt.ylabel('Réalité')
    plt.title('Confusion Matrix')
    
    # Créer le dossier rapports si inexistant
    if not os.path.exists("../rapports"):
        os.makedirs("../rapports")
    plt.savefig("../rapports/confusion_matrix.png")
    print("Confusion matrix sauvegardée dans rapports/confusion_matrix.png")
    
    # 9. Sauvegarder le modèle
    if not os.path.exists("../modele"):
        os.makedirs("../modele")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Modèle sauvegardé dans {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"Erreur lors de l'entraînement: {e}")
