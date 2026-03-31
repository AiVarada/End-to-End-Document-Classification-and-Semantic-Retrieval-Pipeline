# End-to-End Document Classification and Semantic Retrieval Pipeline

An NLP pipeline for multi-class document classification and semantic retrieval over 1,200+ documents. Compares Logistic Regression, SVM, and Naive Bayes using 5-fold stratified cross-validation, achieving 89% weighted F1 with LinearSVC. Extended with a transformer-based RAG component using HuggingFace sentence-transformers and FAISS vector indexing, improving Precision@5 by 38% over a TF-IDF baseline. Deployed on Streamlit with real-time classification, side-by-side retrieval comparison, and LLM answer generation via Groq API.

🔗 **Live App:** [end-to-end-document-classification-and-semantic-retrieval-pipe.streamlit.app](https://end-to-end-document-classification-and-semantic-retrieval-pipe.streamlit.app/)

---

## What This App Does

**Tab 1 — Classify Document**
Upload a document or paste text and the trained model predicts which of 8 categories it belongs to. Shows confidence scores across all categories with a bar chart.

**Tab 2 — Semantic Retrieval**
Enter a natural language query. Both retrievers — FAISS RAG and TF-IDF baseline — run simultaneously and return the most relevant document chunks side by side. The Groq LLM then generates a direct answer using the retrieved chunks as context.

**Tab 3 — Model Evaluation**
Real results from training — classifier comparison table, per-fold F1 chart, confusion matrix, per-class F1 report, and a side-by-side retrieval metric comparison (MRR, P@1, P@5, P@10).

---

## Tech Stack

| Component | Technology |
|---|---|
| Classification | Scikit-learn — LinearSVC, Logistic Regression, Naive Bayes |
| Feature Extraction | TF-IDF (50k features, bigrams, sublinear TF) |
| Embeddings | HuggingFace sentence-transformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS IndexFlatIP (cosine similarity) |
| LLM Generation | Groq API (Llama 3.1-8b-instant) |
| Frontend | Streamlit |
| Dataset | 20 Newsgroups (mapped to 8 categories) |

---

## Dataset

Uses the [20 Newsgroups dataset](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset) loaded directly via scikit-learn. The 20 original categories are mapped to 8 broader categories:

| Category | Original Newsgroups |
|---|---|
| Technology | comp.graphics, comp.sys.*, comp.windows.x, sci.electronics |
| Medical | sci.med |
| Research | alt.atheism, sci.crypt, sci.space |
| Legal | talk.politics.guns, talk.politics.mideast, talk.politics.misc |
| Compliance | soc.religion.christian, talk.religion.misc |
| Operations | misc.forsale, rec.autos, rec.motorcycles, rec.sport.* |

---

## Project Structure
```
├── app.py                          # Streamlit frontend
├── train.py                        # Training pipeline
├── preprocessor.py                 # Shared TextPreprocessor class
├── requirements.txt
├── saved_model/                    # Pre-trained model artifacts
│   ├── best_classifier.pkl         # Best classifier pipeline (SVM)
│   ├── label_encoder.pkl           # Label encoder
│   ├── cv_results.pkl              # Cross-validation results (all 3 models)
│   ├── eval_results.pkl            # Confusion matrix + per-class F1
│   ├── tfidf_retriever.pkl         # TF-IDF vectorizer + matrix
│   ├── faiss_index.bin             # FAISS vector index
│   ├── chunk_metadata.pkl          # Chunk text, labels, embeddings
│   └── retrieval_results.pkl       # MRR + Precision@K results
├── ExampleDocumentsClassify/       # Sample documents to upload and classify
│   ├── technology_graphics_issue.txt
│   ├── medical_medication_question.txt
│   ├── legal_gun_politics.txt
│   ├── research_space_mission.txt
│   ├── operations_forsale_motorcycle.txt
│   └── compliance_religion_discussion.pdf
├── classifyDocumentsExamples.txt   # Text examples to paste and classify
└── semanticRetrievalExamples.txt   # Retrieval queries to try in Tab 2
```

---

## How to Use the App

### Classify a Document (Tab 1)

**Option A — Text Input**
- Select a sample from the dropdown, or type/paste your own text
- See example texts in `classifyDocumentsExamples.txt`
- Click **Classify**

**Option B — Upload a Document**
- Switch to Upload Document mode
- Upload any `.txt` or `.pdf` file
- Ready-to-use example documents are in the `ExampleDocumentsClassify/` folder
- Click **Classify**

### Semantic Retrieval + LLM Answer (Tab 2)

- Enter a natural language query in the search box
- Optionally filter by category
- Click **Search Both Retrievers**
- Both FAISS RAG and TF-IDF results appear side by side
- Enter your Groq API key in the sidebar to get an LLM-generated answer from the retrieved chunks
- See `semanticRetrievalExamples.txt` for example queries that work well

### View Evaluation Results (Tab 3)

- All metrics shown are real results from training — nothing is hardcoded
- Includes classifier comparison, per-fold F1, confusion matrix, per-class report, and retrieval metric comparison

---

## Getting a Groq API Key

The LLM answer generation requires a free Groq API key.

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Generate an API key
4. Paste it into the **API Configuration** field in the sidebar

The key is never stored — it is only used for the current session.

---

## Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The `saved_model/` folder is already included in the repo — no training required to run the app.

---

## Retrain the Model (Optional)

The pre-trained models are already included in `saved_model/`. If you want to retrain from scratch:
```bash
python train.py
```

This will:
- Download the 20 Newsgroups dataset automatically via scikit-learn
- Compare LR, SVM, and Naive Bayes using 5-fold cross-validation
- Build the FAISS index using HuggingFace sentence-transformers
- Evaluate both retrievers using MRR and Precision@K
- Overwrite everything in `saved_model/`

Training takes approximately 5-10 minutes on CPU.

> **Planned Enhancement:** A future version will include a **Train Model** button directly in the app so users can retrain without using the command line.

---


## Pipeline Overview
```
20 Newsgroups Dataset (1,200+ docs, 8 categories)
          │
          ▼
   Text Preprocessing
   (lowercase, clean, strip headers)
          │
          ├──────────────────────────────────┐
          ▼                                  ▼
  Classification Pipeline           RAG Retrieval Pipeline
          │                                  │
   TF-IDF Vectorizer              Chunk documents (~150 words)
          │                                  │
   Compare 3 Models               HuggingFace Embeddings
   LR | SVM | Naive Bayes         (all-MiniLM-L6-v2, 384-dim)
          │                                  │
   5-Fold Stratified CV           FAISS IndexFlatIP
          │                       vs TF-IDF Baseline
   Best Model: SVM                           │
   F1: 89%                        MRR | P@1 | P@5 | P@10
          │                                  │
          └──────────────┬───────────────────┘
                         ▼
                  Streamlit App
          Classify | Retrieve | Evaluate
                         │
                    Groq LLM
               RAG Answer Generation
```

---


## Acknowledgements

- [20 Newsgroups Dataset](https://scikit-learn.org/stable/datasets/real_world.html) via scikit-learn
- [sentence-transformers](https://www.sbert.net/) by UKPLab
- [FAISS](https://github.com/facebookresearch/faiss) by Meta AI
- [Groq](https://groq.com/) for fast LLM inference
- [Streamlit](https://streamlit.io/) for the frontend
