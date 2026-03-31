"""
train.py — End-to-End NLP Pipeline
====================================
Matches project description exactly:
  - Multi-class document classification (1,000+ docs)
  - Systematic model comparison: Logistic Regression, SVM, Naive Bayes
  - 5-fold cross-validation, F1 scoring
  - Transformer-based RAG component (HuggingFace sentence-transformers)
  - FAISS vector indexing
  - TF-IDF baseline retrieval
  - MRR and Precision@K evaluation
  - Saves everything for app.py to load

Usage:
    python train.py

Saves to ./saved_model/:
    best_classifier.pkl     Best classifier pipeline (SVM/LR/NB)
    label_encoder.pkl       Label encoder
    cv_results.pkl          All 3 classifiers CV results (for app.py Tab 3)
    tfidf_retriever.pkl     TF-IDF vectorizer for baseline retrieval
    faiss_index.bin         FAISS vector index (transformer embeddings)
    chunk_metadata.pkl      Chunk text + labels
    retrieval_results.pkl   MRR + Precision@K comparison (FAISS vs TF-IDF)
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import classification_report,confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from preprocessor import TextPreprocessor

warnings.filterwarnings("ignore")

SAVE_DIR = "./saved_model"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Category mapping: 20 Newsgroups → 8 categories ───────────────────────────
CATEGORY_MAP = { # We can change the way we categories these
    "alt.atheism":               "Research",
    "comp.graphics":             "Technology",
    "comp.os.ms-windows.misc":   "Technology",
    "comp.sys.ibm.pc.hardware":  "Technology",
    "comp.sys.mac.hardware":     "Technology",
    "comp.windows.x":            "Technology",
    "misc.forsale":              "Operations",
    "rec.autos":                 "Operations",
    "rec.motorcycles":           "Operations",
    "rec.sport.baseball":        "Operations",
    "rec.sport.hockey":          "Operations",
    "sci.crypt":                 "Research",
    "sci.electronics":           "Technology",
    "sci.med":                   "Medical",
    "sci.space":                 "Research",
    "soc.religion.christian":    "Compliance",
    "talk.politics.guns":        "Legal",
    "talk.politics.mideast":     "Legal",
    "talk.politics.misc":        "Legal",
    "talk.religion.misc":        "Compliance",
}

# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load & Prepare Data
# ════════════════════════════════════════════════════════════════════════════
def load_data(max_per_class=150):
    print("\n" + "="*60)
    print("  STEP 1: Loading 20 Newsgroups Dataset")
    print("="*60)

    raw = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))

    by_class = defaultdict(list)
    for text, target in zip(raw.data, raw.target):
        label = CATEGORY_MAP[raw.target_names[target]]
        if len(text.strip()) > 100:
            by_class[label].append(text.strip())

    texts, labels = [], []
    for label, docs in by_class.items():
        sampled = docs[:max_per_class]
        texts.extend(sampled)
        labels.extend([label] * len(sampled))

    print(f"  Total documents : {len(texts)}")
    print(f"  Categories      : {len(set(labels))}")
    print()
    for lbl in sorted(set(labels)):
        print(f"    {lbl:<15} {labels.count(lbl)} docs")

    return texts, labels

# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Classification: Compare LR vs SVM vs Naive Bayes (5-Fold CV)
# ════════════════════════════════════════════════════════════════════════════
def compare_classifiers(texts, labels):
    print("\n" + "="*60)
    print("  STEP 2: Classifier Comparison — 5-Fold Cross Validation")
    print("="*60)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", multi_class="multinomial"),
        "SVM (LinearSVC)":     LinearSVC(max_iter=2000, C=1.0),
        "Naive Bayes":         MultinomialNB(alpha=0.1),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}

    for name, clf in classifiers.items():
        print(f"\n  Training {name}...")
        pipeline = Pipeline([
            ("preprocessor", TextPreprocessor()),
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", clf),
        ])
        scores = cross_validate(
            pipeline, texts, y, cv=skf,
            scoring=["f1_weighted", "precision_weighted", "recall_weighted"],
            n_jobs=-1
        )
        result = {
            "F1 Score":  scores["test_f1_weighted"].mean(),
            "Precision": scores["test_precision_weighted"].mean(),
            "Recall":    scores["test_recall_weighted"].mean(),
            "F1 Std":    scores["test_f1_weighted"].std(),
            "Fold F1s":  scores["test_f1_weighted"].tolist(),
        }
        cv_results[name] = result
        print(f"    F1:        {result['F1 Score']:.4f} ± {result['F1 Std']:.4f}")
        print(f"    Precision: {result['Precision']:.4f}")
        print(f"    Recall:    {result['Recall']:.4f}")

    # Print comparison table
    print("\n  " + "-"*50)
    print(f"  {'Model':<22} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print("  " + "-"*50)
    for name, r in cv_results.items():
        print(f"  {name:<22} {r['F1 Score']:>8.4f} {r['Precision']:>10.4f} {r['Recall']:>8.4f}")
    print("  " + "-"*50)

    # Pick best model by F1
    best_name = max(cv_results, key=lambda k: cv_results[k]["F1 Score"])
    print(f"\n  Best model: {best_name} (F1 = {cv_results[best_name]['F1 Score']:.4f})")

    # Train best model on full data and save
    best_clf = classifiers[best_name]
    best_pipeline = Pipeline([
        ("preprocessor", TextPreprocessor()),
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", best_clf),
    ])
    best_pipeline.fit(texts, y)
    
    # Confusion Matrix
    X_train, X_test, y_train, y_test = train_test_split(
    texts, y, test_size=0.2, random_state=42, stratify=y)

    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

    # Save alongside cv_results
    with open(f"{SAVE_DIR}/eval_results.pkl", "wb") as f:
        pickle.dump({"confusion_matrix": cm,
                    "class_report": report,
                    "labels": le.classes_.tolist()}, f)

        with open(f"{SAVE_DIR}/best_classifier.pkl", "wb") as f:
            pickle.dump(best_pipeline, f)
        with open(f"{SAVE_DIR}/label_encoder.pkl", "wb") as f:
            pickle.dump(le, f)
        with open(f"{SAVE_DIR}/cv_results.pkl", "wb") as f:
            pickle.dump({"results": cv_results, "best_model": best_name}, f)

        print(f"  Saved best classifier → {SAVE_DIR}/best_classifier.pkl")
        
        return best_pipeline, le, cv_results, best_name


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — Chunk Documents for Retrieval
# ════════════════════════════════════════════════════════════════════════════
def chunk_documents(texts, labels, chunk_size=150, overlap=30):
    print("\n" + "="*60)
    print("  STEP 3: Chunking Documents for Retrieval")
    print("="*60)

    all_chunks, all_labels = [], []
    for text, label in zip(texts, labels):
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.strip()) > 50:
                all_chunks.append(chunk)
                all_labels.append(label)

    print(f"  Documents  : {len(texts)}")
    print(f"  Chunks     : {len(all_chunks)}")
    print(f"  Chunk size : ~{chunk_size} words with {overlap}-word overlap")
    return all_chunks, all_labels


# ════════════════════════════════════════════════════════════════════════════
# STEP 4a — TF-IDF Baseline Retrieval
# ════════════════════════════════════════════════════════════════════════════
def build_tfidf_retriever(chunks, labels):
    print("\n" + "="*60)
    print("  STEP 4a: Building TF-IDF Baseline Retriever")
    print("="*60)

    tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True)
    X = tfidf.fit_transform(chunks)

    with open(f"{SAVE_DIR}/tfidf_retriever.pkl", "wb") as f:
        pickle.dump({"vectorizer": tfidf, "matrix": X, "labels": labels, "chunks": chunks}, f)

    print(f"  TF-IDF matrix shape : {X.shape}")
    print(f"  Saved → {SAVE_DIR}/tfidf_retriever.pkl")
    return tfidf, X


# ════════════════════════════════════════════════════════════════════════════
# STEP 4b — Transformer Embeddings + FAISS Index (RAG)
# ════════════════════════════════════════════════════════════════════════════
def build_faiss_index(chunks, labels):
    print("\n" + "="*60)
    print("  STEP 4b: Building FAISS Index (HuggingFace Transformers)")
    print("="*60)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  ERROR: pip install sentence-transformers")
        raise

    print("  Loading HuggingFace model: all-MiniLM-L6-v2 (384-dim)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"  Encoding {len(chunks)} chunks...")
    embeddings = model.encode(
        chunks,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True  # cosine similarity via inner product
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product on normalized = cosine
    index.add(embeddings)

    faiss.write_index(index, f"{SAVE_DIR}/faiss_index.bin")
    with open(f"{SAVE_DIR}/chunk_metadata.pkl", "wb") as f:
        pickle.dump({"chunks": chunks, "labels": labels, "embeddings": embeddings}, f)

    print(f"  FAISS index : {index.ntotal} vectors, dim={dim}")
    print(f"  Saved → {SAVE_DIR}/faiss_index.bin")
    return index, embeddings


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — Evaluate Both Retrievers: MRR and Precision@K
# ════════════════════════════════════════════════════════════════════════════
def evaluate_retrievers(chunks, labels, tfidf_vec, tfidf_matrix, faiss_index, faiss_embeddings, n_queries=100):
    print("\n" + "="*60)
    print("  STEP 5: Retrieval Evaluation — MRR and Precision@K")
    print("="*60)

    label_arr = np.array(labels)
    query_indices = np.random.RandomState(42).choice(len(chunks), size=min(n_queries, len(chunks)), replace=False)

    def compute_metrics(retrieved_labels_list, true_labels, k_values=[1, 5, 10]):
        mrr_scores = []
        pk_scores = {k: [] for k in k_values}

        for retrieved, true_label in zip(retrieved_labels_list, true_labels):
            # MRR — rank of first correct result
            rr = 0
            for rank, lbl in enumerate(retrieved, 1):
                if lbl == true_label:
                    rr = 1 / rank
                    break
            mrr_scores.append(rr)

            # Precision@K
            for k in k_values:
                top_k = retrieved[:k]
                pk_scores[k].append(sum(l == true_label for l in top_k) / k)

        return {
            "MRR": np.mean(mrr_scores),
            **{f"P@{k}": np.mean(pk_scores[k]) for k in k_values}
        }

    # ── TF-IDF Retrieval ──────────────────────────────────────────────────
    print("\n  Evaluating TF-IDF baseline...")
    tfidf_retrieved = []
    for qi in query_indices:
        q_vec = tfidf_matrix[qi]
        sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
        sims[qi] = -1  # exclude self
        top_idx = np.argsort(sims)[::-1][:10]
        tfidf_retrieved.append(label_arr[top_idx].tolist())

    tfidf_metrics = compute_metrics(tfidf_retrieved, label_arr[query_indices])
    print(f"    MRR  : {tfidf_metrics['MRR']:.4f}")
    print(f"    P@1  : {tfidf_metrics['P@1']:.4f}")
    print(f"    P@5  : {tfidf_metrics['P@5']:.4f}")
    print(f"    P@10 : {tfidf_metrics['P@10']:.4f}")

    # ── FAISS RAG Retrieval ───────────────────────────────────────────────
    print("\n  Evaluating FAISS + Transformer RAG...")
    faiss_retrieved = []
    for qi in query_indices:
        q_emb = faiss_embeddings[qi].reshape(1, -1)
        scores, indices = faiss_index.search(q_emb, 11)  # +1 to exclude self
        top_idx = [idx for idx in indices[0] if idx != qi][:10]
        faiss_retrieved.append(label_arr[top_idx].tolist())

    faiss_metrics = compute_metrics(faiss_retrieved, label_arr[query_indices])
    print(f"    MRR  : {faiss_metrics['MRR']:.4f}")
    print(f"    P@1  : {faiss_metrics['P@1']:.4f}")
    print(f"    P@5  : {faiss_metrics['P@5']:.4f}")
    print(f"    P@10 : {faiss_metrics['P@10']:.4f}")

    # ── Comparison ────────────────────────────────────────────────────────
    print("\n  " + "-"*50)
    print(f"  {'Metric':<10} {'TF-IDF':>12} {'FAISS+RAG':>12} {'Improvement':>12}")
    print("  " + "-"*50)
    for metric in ["MRR", "P@1", "P@5", "P@10"]:
        tfidf_val = tfidf_metrics[metric]
        faiss_val = faiss_metrics[metric]
        improvement = ((faiss_val - tfidf_val) / tfidf_val) * 100
        print(f"  {metric:<10} {tfidf_val:>12.4f} {faiss_val:>12.4f} {improvement:>11.1f}%")
    print("  " + "-"*50)

    retrieval_results = {
        "tfidf": tfidf_metrics,
        "faiss": faiss_metrics,
        "n_queries": len(query_indices),
    }
    with open(f"{SAVE_DIR}/retrieval_results.pkl", "wb") as f:
        pickle.dump(retrieval_results, f)

    print(f"\n  Saved → {SAVE_DIR}/retrieval_results.pkl")
    return retrieval_results


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  INTELLIGENT DOCUMENT CLASSIFICATION & RETRIEVAL")
    print("  End-to-End NLP Pipeline")
    print("="*60)

    # Step 1 — Load data
    texts, labels = load_data(max_per_class=150)  # ~1,200 docs total

    # Step 2 — Compare classifiers (LR vs SVM vs NB), 5-fold CV
    best_pipeline, le, cv_results, best_name = compare_classifiers(texts, labels)

    # Step 3 — Chunk documents for retrieval
    chunks, chunk_labels = chunk_documents(texts, labels)

    # Step 4a — TF-IDF baseline retriever
    tfidf_vec, tfidf_matrix = build_tfidf_retriever(chunks, chunk_labels)

    # Step 4b — HuggingFace transformer embeddings + FAISS index
    faiss_index, faiss_embeddings = build_faiss_index(chunks, chunk_labels)

    # Step 5 — Evaluate both retrievers: MRR, P@1, P@5, P@10
    retrieval_results = evaluate_retrievers(
        chunks, chunk_labels,
        tfidf_vec, tfidf_matrix,
        faiss_index, faiss_embeddings
    )

    # ── Final Summary ─────────────────────────────────────────────────────
    best_f1 = cv_results[best_name]["F1 Score"]
    p5_tfidf = retrieval_results["tfidf"]["P@5"]
    p5_faiss = retrieval_results["faiss"]["P@5"]
    improvement = ((p5_faiss - p5_tfidf) / p5_tfidf) * 100

    print("\n" + "="*60)
    print("  TRAINING COMPLETE — SUMMARY")
    print("="*60)
    print(f"  Best Classifier    : {best_name}")
    print(f"  F1 Score (5-CV)    : {best_f1:.4f}")
    print(f"  TF-IDF  P@5        : {p5_tfidf:.4f}")
    print(f"  FAISS   P@5        : {p5_faiss:.4f}")
    print(f"  Retrieval Gain     : +{improvement:.1f}%")
    print(f"  Saved to           : {SAVE_DIR}/")
    print("="*60)
    print("\n  Next: streamlit run app.py\n")
