# Product Review Intelligence 🤖

## Project Description
This university project consists of an AI multi-agent system designed to analyze Amazon product reviews. It uses a combination of Deep Learning (BERT) for precise sentiment analysis and AI agents (CrewAI + Ollama/llama3.2 (3B)) for market analysis and recommendation generation.

## System Architecture
1. **Agent 1 (Sentiment Analyst)**: Uses a fine-tuned BERT model to classify reviews as Positive, Negative, or Neutral.
2. **Agent 2 (Market Researcher)**: Analyzes Agent 1's results via Ollama/llama3.2 (3B) to extract strengths/weaknesses and suggest improvements.
3. **Orchestrator**: Coordinates agents and manages the Human-In-The-Loop (HITL) validation process.
4. **Gradio Interface**: Provides a modern and interactive web interface.

## Prerequisites
- Python 3.10+
- Ollama installed and running (with the llama3.2 model: `ollama pull llama3.2`)
- Amazon Dataset (`7817_1.csv`) in the `dataset/` folder

## Installation
1. Clone the project:
```bash
git clone <repository-url>
cd projet-avis-produits
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration
- Place your `7817_1.csv` file in the `dataset/` folder.
- Ensure Ollama is running on `http://localhost:11434`.

## Usage

### 1. Training the BERT Model
This step was quickly performed on Google Colab using hardware acceleration (GPU T4). 
**Note:** This step is **optional for local jury evaluation**, as the pre-trained model folder (`modele/bert_model_complet/`) is already provided in the repository to avoid long waiting times on a standard processor (CPU).

If you still wish to restart the training:
```bash
cd colab
python train_bert.py
```
This will generate the `modele/bert_model_complet/` folder and a confusion matrix in `rapports/`.

### 2. Launching the Web Application
```bash
cd frontend
python app.py
```
The interface will be accessible locally and via a Gradio public link (if `share=True`).

## File Structure
- `backend/`: Business logic, CrewAI agents, and BERT model management.
- `frontend/`: Gradio user interface.
- `colab/`: Training script for the BERT model.
- `dataset/`: Contains the source CSV file.
- `modele/`: Stores the trained model (`bert_model_complet/` folder generated via `save_pretrained()`).
- `logs/`: Traces of all system actions in JSON format.
- `rapports/`: Generated PDF reports and visualizations.

## Dataset
The project uses the "Amazon Product Reviews" dataset (7817_1.csv). The columns used are `reviews.text` for analysis and `reviews.rating` for labeling during training.

## Authors
Ikram OIALILI & Mohamed Taha DOUMI.
Project realized as part of the university curriculum at UIR.
