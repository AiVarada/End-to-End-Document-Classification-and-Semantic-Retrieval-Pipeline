"""
app.py — Streamlit Frontend
============================
Loads real trained artifacts from train.py and exposes:
  Tab 1 — Document Classification (real model inference)
  Tab 2 — Semantic Retrieval: FAISS RAG vs TF-IDF (both methods, side by side)
  Tab 3 — Model Evaluation: Classifier comparison + Retrieval MRR/P@K results

Run: streamlit run app.py
"""

import os
import pickle
import numpy as np
import streamlit as st
import pandas as pd
import faiss
import pypdf
from preprocessor import TextPreprocessor

SAVE_DIR = "./saved_model"

st.set_page_config(page_title="DocIntel", page_icon="🧠", layout="wide")
st.title("🧠 Intelligent Document Classification & Retrieval")
st.caption("Scikit-learn · HuggingFace Transformers · FAISS · RAG · 5-Fold CV · MRR · Precision@K")

# ── Guard: must run train.py first ───────────────────────────────────────────
required = ["best_classifier.pkl", "label_encoder.pkl", "cv_results.pkl",
            "tfidf_retriever.pkl", "faiss_index.bin",
            "chunk_metadata.pkl", "retrieval_results.pkl"]

missing = [f for f in required if not os.path.exists(f"{SAVE_DIR}/{f}")]
if missing:
    st.error("⚠️ Trained model not found. Run training first.")
    st.code("python train.py", language="bash")
    st.write("**Missing files:**")
    for m in missing:
        st.write(f"  - `{SAVE_DIR}/{m}`")
    st.stop()

# ── Load all artifacts ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model and indexes...")
def load_all():
    with open(f"{SAVE_DIR}/best_classifier.pkl", "rb") as f:
        classifier = pickle.load(f)
    with open(f"{SAVE_DIR}/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open(f"{SAVE_DIR}/cv_results.pkl", "rb") as f:
        cv_data = pickle.load(f)
    with open(f"{SAVE_DIR}/tfidf_retriever.pkl", "rb") as f:
        tfidf_data = pickle.load(f)
    with open(f"{SAVE_DIR}/chunk_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    with open(f"{SAVE_DIR}/retrieval_results.pkl", "rb") as f:
        retrieval_results = pickle.load(f)

    faiss_index = faiss.read_index(f"{SAVE_DIR}/faiss_index.bin")

    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        embedder = None

    return {
        "classifier": classifier,
        "le": le,
        "cv_data": cv_data,
        "tfidf_vec": tfidf_data["vectorizer"],
        "tfidf_matrix": tfidf_data["matrix"],
        "tfidf_chunks": tfidf_data["chunks"],
        "tfidf_labels": tfidf_data["labels"],
        "faiss_index": faiss_index,
        "faiss_embeddings": metadata["embeddings"],
        "chunks": metadata["chunks"],
        "chunk_labels": metadata["labels"],
        "retrieval_results": retrieval_results,
        "embedder": embedder,
    }

R = load_all()

# ── Inference helpers ─────────────────────────────────────────────────────────
def classify_text(text):
    le = R["le"]
    clf = R["classifier"]
    pred_idx = clf.predict([text])[0]
    predicted = le.inverse_transform([pred_idx])[0]
    decision = clf.decision_function([text])[0]
    exp_d = np.exp(decision - decision.max())
    probs = exp_d / exp_d.sum()
    scores = {le.classes_[i]: float(probs[i]) for i in range(len(le.classes_))}
    return predicted, scores


def retrieve_tfidf(query, top_k=5, filter_cat=None):
    from sklearn.metrics.pairwise import cosine_similarity
    tfidf_vec = R["tfidf_vec"]
    tfidf_matrix = R["tfidf_matrix"]
    chunks = R["tfidf_chunks"]
    labels = R["tfidf_labels"]

    # Pre-filter indices by category
    if filter_cat:
        indices = [i for i, l in enumerate(labels) if l in filter_cat]
    else:
        indices = list(range(len(chunks)))

    filtered_matrix = tfidf_matrix[indices]
    q_vec = tfidf_vec.transform([query])
    sims = cosine_similarity(q_vec, filtered_matrix).flatten()
    top_local = np.argsort(sims)[::-1][:top_k]
    top_idx = [indices[i] for i in top_local]
    return [{"chunk": chunks[i], "label": labels[i], "score": float(sims[top_local[j]])}
            for j, i in enumerate(top_idx)]


def retrieve_faiss(query, top_k=5, filter_cat=None):
    embedder = R["embedder"]
    if embedder is None:
        return []

    chunks = R["chunks"]
    labels = R["chunk_labels"]
    all_embeddings = R["faiss_embeddings"]

    # Pre-filter indices by category
    if filter_cat:
        indices = [i for i, l in enumerate(labels) if l in filter_cat]
    else:
        indices = list(range(len(chunks)))

    # Build a temporary FAISS index from filtered embeddings only
    filtered_embs = all_embeddings[indices].astype("float32")
    dim = filtered_embs.shape[1]
    temp_index = faiss.IndexFlatIP(dim)
    temp_index.add(filtered_embs)

    q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, local_idx = temp_index.search(q_emb, top_k)

    return [
        {"chunk": chunks[indices[i]], "label": labels[indices[i]], "score": float(scores[0][j])}
        for j, i in enumerate(local_idx[0]) if i != -1
    ]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    with st.sidebar:
        st.header("🔑 API Configuration")
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="Enter your Groq API key...",
            help="Get a free key at console.groq.com"
        )
        st.divider()
    st.header("📊 Dataset")
    st.metric("Source", "20 Newsgroups")
    st.metric("Documents", "1,200+")
    st.metric("Categories", str(len(R["le"].classes_)))
    st.metric("FAISS Vectors", str(R["faiss_index"].ntotal))

    st.divider()
    st.header("🏆 Best Classifier")
    best_name = R["cv_data"]["best_model"]
    best_f1 = R["cv_data"]["results"][best_name]["F1 Score"]
    st.metric("Model", best_name)
    st.metric("F1 Score (5-CV)", f"{best_f1:.4f}")

    st.divider()
    st.header("🏷️ Categories")
    for cat in sorted(R["le"].classes_):
        st.write(f"• {cat}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📄 Classify Document",
    "🔍 Semantic Retrieval",
    "📊 Model Evaluation"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Document Classification
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Document Classification")
    st.write(f"Using real trained **{best_name}** model. Paste any text to classify.")

    SAMPLES = {
        "Medical": "The patient presented with acute chest pain and elevated troponin levels. ECG showed ST elevation in leads II, III, aVF. Administered aspirin 325mg and referred to cardiology for emergency catheterization.",
        "Technology": "The new GPU architecture introduces tensor cores optimized for transformer inference. Memory bandwidth increased by 40% over the previous generation enabling faster LLM inference.",
        "Legal": "The defendant hereby agrees to indemnify and hold harmless the plaintiff from any claims arising from the breach of contract. Jurisdiction shall be the state of California with binding arbitration.",
        "Finance": "Q3 earnings beat estimates by 12 cents per share. Revenue grew 18% year-over-year to $4.2B. Operating margins expanded 200bps. The board approved a $500M share buyback program.",
        "Research": "We propose a novel attention mechanism that reduces quadratic complexity to linear. Experiments on GLUE benchmark show 3% improvement over BERT-base with 40% fewer parameters.",
    }

    input_mode = st.radio("Input Method", ["Text Input", "Upload Document"], horizontal=True)

    doc_text = ""

    if input_mode == "Text Input":
        sample_choice = st.selectbox(
            "Load a sample document",
            ["-- Type your own --"] + list(SAMPLES.keys())
        )
        default_text = SAMPLES[sample_choice] if sample_choice != "-- Type your own --" else ""
        doc_text = st.text_area("Document Text", value=default_text, height=200,
                                placeholder="Paste your document text here...")

    elif input_mode == "Upload Document":
        uploaded = st.file_uploader("Upload a document", type=["txt", "pdf"])
        if uploaded is not None:
            if uploaded.type == "text/plain":
                doc_text = uploaded.read().decode("utf-8")
            elif uploaded.type == "application/pdf":
                import pypdf
                reader = pypdf.PdfReader(uploaded)
                doc_text = " ".join(page.extract_text() for page in reader.pages)

            # Show preview of extracted text
            if doc_text:
                st.success(f"Document loaded: {uploaded.name}")
                with st.expander("Preview extracted text"):
                    st.write(doc_text[:1000] + ("..." if len(doc_text) > 1000 else ""))
        else:
            st.info("Upload a .txt or .pdf file to classify.")

    if st.button("⚡ Classify", type="primary"):
        if not doc_text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Running inference..."):
                predicted, scores = classify_text(doc_text)

            st.success(f"✅ Predicted: **{predicted}**")
            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Category", predicted)
                st.metric("Confidence", f"{scores[predicted]*100:.1f}%")
                st.caption(f"Model: {best_name}")

            with col2:
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                conf_df = pd.DataFrame(sorted_scores, columns=["Category", "Confidence"])
                conf_df["Confidence %"] = (conf_df["Confidence"] * 100).round(2)
                st.dataframe(conf_df[["Category", "Confidence %"]],
                             use_container_width=True, hide_index=True)

            st.subheader("Confidence Distribution")
            chart_df = pd.DataFrame(sorted_scores, columns=["Category", "Score"]).set_index("Category")
            st.bar_chart(chart_df)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Semantic Retrieval: FAISS RAG vs TF-IDF (side by side)
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Semantic Retrieval — FAISS RAG vs TF-IDF Baseline")
    st.write("Both retrievers run on the same query. Compare results side by side.")

    if R["embedder"] is None:
        st.warning("⚠️ sentence-transformers not installed. FAISS retrieval unavailable.\n"
                   "Run: `pip install sentence-transformers`")

    query = st.text_input("Search Query", placeholder="e.g. neural network GPU memory bandwidth")
    top_k = st.slider("Top-K Results", 1, 10, 5)

    col_f, col_t = st.columns(2)
    with col_f:
        filter_cat = st.multiselect("Filter category", sorted(R["le"].classes_), key="filter1")
    with col_t:
        search_btn = st.button("🔍 Search Both Retrievers", type="primary")

    if search_btn:
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Running both retrievers..."):
                from sklearn.metrics.pairwise import cosine_similarity as cos_sim

                faiss_results = retrieve_faiss(query, top_k=top_k, filter_cat=filter_cat if filter_cat else None)
                tfidf_results = retrieve_tfidf(query, top_k=top_k, filter_cat=filter_cat if filter_cat else None)

            col_left, col_right = st.columns(2)

            # ── FAISS RAG Results ─────────────────────────────────────────
            with col_left:
                st.markdown("#### 🚀 FAISS + Transformer RAG")
                st.caption("HuggingFace all-MiniLM-L6-v2 embeddings · Cosine similarity")
                if not faiss_results:
                    st.warning("sentence-transformers not installed.")
                else:
                    for i, r in enumerate(faiss_results):
                        with st.expander(f"#{i+1} [{r['label']}]  Score: {r['score']:.4f}", expanded=i < 2):
                            st.metric("Similarity Score", f"{r['score']:.4f}")
                            st.metric("Category", r["label"])
                            st.write(r["chunk"][:400] + "...")

            # ── TF-IDF Results ────────────────────────────────────────────
            with col_right:
                st.markdown("#### 📊 TF-IDF Baseline")
                st.caption("Bag-of-words · Cosine similarity on TF-IDF vectors")
                for i, r in enumerate(tfidf_results):
                    with st.expander(f"#{i+1} [{r['label']}]  Score: {r['score']:.4f}", expanded=i < 2):
                        st.metric("Similarity Score", f"{r['score']:.4f}")
                        st.metric("Category", r["label"])
                        st.write(r["chunk"][:400] + "...")

            # ── Score comparison table ────────────────────────────────────
            st.divider()
            st.subheader("Score Comparison")
            max_len = max(len(faiss_results), len(tfidf_results))
            comparison = []
            for i in range(max_len):
                row = {"Rank": i + 1}
                if i < len(faiss_results):
                    row["FAISS Category"] = faiss_results[i]["label"]
                    row["FAISS Score"] = round(faiss_results[i]["score"], 4)
                if i < len(tfidf_results):
                    row["TF-IDF Category"] = tfidf_results[i]["label"]
                    row["TF-IDF Score"] = round(tfidf_results[i]["score"], 4)
                comparison.append(row)
            st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)

            # ── pip install groq ────────────────────────────────────
            from groq import Groq
            if not groq_api_key:
                st.warning("Enter your Groq API key in the sidebar to enable answer generation.")
            else:
                client = Groq(api_key=groq_api_key)

            if faiss_results:
                context = "\n\n".join([r["chunk"] for r in faiss_results[:3]])
                prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer based on context only:"

                st.write("#### Generated Answer (RAG)")
                with st.spinner("Generating answer..."):
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    st.write(response.choices[0].message.content)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Evaluation (real results from train.py)
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Evaluation — Real Results from Training")

    # ── Classifier Comparison ─────────────────────────────────────────────
    st.write("### Classifier Comparison — 5-Fold Cross Validation")
    st.caption("Logistic Regression vs SVM vs Naive Bayes — trained on 20 Newsgroups (1,200+ docs)")

    cv_results = R["cv_data"]["results"]
    best_model = R["cv_data"]["best_model"]

    clf_rows = []
    for name, r in cv_results.items():
        clf_rows.append({
            "Model": name + (" ✅ best" if name == best_model else ""),
            "F1 Score": round(r["F1 Score"], 4),
            "Precision": round(r["Precision"], 4),
            "Recall": round(r["Recall"], 4),
            "F1 Std (±)": round(r["F1 Std"], 4),
        })
    clf_df = pd.DataFrame(clf_rows)
    st.dataframe(clf_df, use_container_width=True, hide_index=True)

    # Load eval_results.pkl
    with open(f"{SAVE_DIR}/eval_results.pkl", "rb") as f:
        eval_data = pickle.load(f)

    # Confusion matrix as dataframe
    cm_df = pd.DataFrame(eval_data["confusion_matrix"],
            index=eval_data["labels"],
            columns=eval_data["labels"])
    st.write("#### Confusion Matrix")
    st.dataframe(cm_df, use_container_width=True)

    # Per-class F1
    report = eval_data["class_report"]
    rows = []
    for label in eval_data["labels"]:
        rows.append({
            "Category": label,
            "F1": round(report[label]["f1-score"], 3),
            "Precision": round(report[label]["precision"], 3),
            "Recall": round(report[label]["recall"], 3),
            "Support": int(report[label]["support"]),
        })
    st.write("#### Per-Class Report")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Per-fold F1 chart
    st.write("#### Per-Fold F1 Scores")
    fold_data = {}
    for name, r in cv_results.items():
        fold_data[name] = r["Fold F1s"]
    fold_df = pd.DataFrame(fold_data, index=[f"Fold {i+1}" for i in range(5)])
    st.line_chart(fold_df)

    st.divider()

    # ── Retrieval Comparison: MRR + Precision@K ───────────────────────────
    st.write("### Retrieval Evaluation — FAISS RAG vs TF-IDF Baseline")
    st.caption("Evaluated on real queries using MRR and Precision@K metrics")

    rr = R["retrieval_results"]
    tfidf_m = rr["tfidf"]
    faiss_m = rr["faiss"]

    metrics = ["MRR", "P@1", "P@5", "P@10"]
    retrieval_rows = []
    for metric in metrics:
        tval = tfidf_m[metric]
        fval = faiss_m[metric]
        improvement = ((fval - tval) / tval) * 100
        retrieval_rows.append({
            "Metric": metric,
            "TF-IDF Baseline": round(tval, 4),
            "FAISS + RAG": round(fval, 4),
            "Improvement": f"+{improvement:.1f}%",
        })

    retrieval_df = pd.DataFrame(retrieval_rows)
    st.dataframe(retrieval_df, use_container_width=True, hide_index=True)

    # Bar chart comparison
    st.write("#### FAISS RAG vs TF-IDF — Score Comparison")
    chart_data = pd.DataFrame({
        "TF-IDF": [tfidf_m[m] for m in metrics],
        "FAISS+RAG": [faiss_m[m] for m in metrics],
    }, index=metrics)
    st.bar_chart(chart_data)

    # Key metrics
    st.divider()
    st.write("### Key Results Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Classifier", best_model.split()[0])
    col2.metric("F1 Score (5-CV)", f"{cv_results[best_model]['F1 Score']:.4f}")
    col3.metric("FAISS MRR", f"{faiss_m['MRR']:.4f}", delta=f"+{((faiss_m['MRR']-tfidf_m['MRR'])/tfidf_m['MRR'])*100:.1f}% vs TF-IDF")
    col4.metric("FAISS P@5", f"{faiss_m['P@5']:.4f}", delta=f"+{((faiss_m['P@5']-tfidf_m['P@5'])/tfidf_m['P@5'])*100:.1f}% vs TF-IDF")

    st.divider()
    st.write("### Pipeline Architecture")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Classification Pipeline**")
        for step, desc in [
            ("1. Data", "20 Newsgroups → 8 categories, 1,200+ docs"),
            ("2. Features", "TF-IDF (50k features, bigrams, sublinear TF)"),
            ("3. Models", "LR · SVM · Naive Bayes compared"),
            ("4. Validation", "5-Fold Stratified Cross-Validation"),
            ("5. Metric", "Weighted F1, Precision, Recall"),
        ]:
            st.markdown(f"**{step}** — {desc}")
    with col2:
        st.write("**RAG Retrieval Pipeline**")
        for step, desc in [
            ("1. Chunking", "~150-word overlapping chunks (30-word overlap)"),
            ("2. Embedding", "HuggingFace all-MiniLM-L6-v2 (384-dim)"),
            ("3. Indexing", "FAISS IndexFlatIP (cosine similarity)"),
            ("4. Baseline", "TF-IDF cosine similarity for comparison"),
            ("5. Metrics", "MRR · Precision@1 · P@5 · P@10"),
        ]:
            st.markdown(f"**{step}** — {desc}")

    st.divider()
    st.write("### How to Reproduce")
    st.code("""
# 1. Install
pip install streamlit scikit-learn faiss-cpu sentence-transformers pandas numpy

# 2. Train (runs all 3 classifiers + builds FAISS index + evaluates both retrievers)
python train.py
(OR)
./venv/bin/python train.py 

# 3. Launch app
streamlit run app.py
    """, language="bash")
