# Nova Reader — RAG Streamlit sur AWS Bedrock

**Résumé**  
Nova Reader est une application web interactive, développée avec *Streamlit*, qui met en œuvre une architecture de *Retrieval-Augmented Generation* (RAG) en partie hébergée sur **AWS Bedrock**.  
L’outil permet d’interroger de manière intelligente des documents PDF ou Word, en combinant recherche sémantique (FAISS + Titan Embeddings) et génération de texte par un modèle de langage (Mistral Large).

---

## 1. Contexte et objectifs

La prolifération de documents non structurés nécessite des outils capables de :
- **Indexer** efficacement du contenu hétérogène (PDF, DOCX),
- **Extraire** l’information pertinente selon une requête,
- **Générer** une réponse cohérente et contextualisée.

Nova Reader répond à ce besoin en offrant :
- une interface conviviale de type *chat*,
- un pipeline complet d’ingestion, de vectorisation et de question-réponse,

---

## 2. Architecture du système

### 2.1 Schéma global

<img width="748" height="564" alt="image" src="https://github.com/user-attachments/assets/2088ba14-f515-4f43-8f9c-877dd7a59986" />

### 2.2 Composants principaux
- **AWS Bedrock**  
  - `amazon.titan-embed-text-v2:0` pour les vecteurs d’embedding  
  - `mistral.mistral-large-2402-v1:0` pour la génération
- **FAISS** : base vectorielle locale pour la similarité
- **LangChain** : gestion des chaînes RAG et du prompt
- **Streamlit** : interface web interactive

---

## 3. Fonctionnalités

- **Téléversement multi-format** : PDF et DOCX (conversion .doc → .docx via Pandoc)  
- **Nettoyage & segmentation**  
  - Suppression du bruit  
  - Découpage par titres, chapitres ou numérotation  
- **Chunking**  
  - *Structured* (taille fixe avec chevauchement)  
  - *Semantic* (clustering hiérarchique avec AgglomerativeClustering)  
- **Recherche vectorielle** : index FAISS alimenté par Titan Embeddings  
- **Question-Réponse** : modèle Mistral Large interrogé via `RetrievalQA`  
- **Interface Chat** : historique de conversation et affichage des sources (section + page)

---

## 4. Installation

### 4.1 Prérequis
- Compte AWS avec accès **Bedrock** (Titan + Mistral)
- `aws-cli` configuré (variables d’environnement ou profil IAM)
- Python ≥ 3.10 et pip

### 4.2 Étapes

```bash
# Cloner le dépôt
git clone https://github.com/Maxcym/RAG_AWS

# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate   # sous Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
