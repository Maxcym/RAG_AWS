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


