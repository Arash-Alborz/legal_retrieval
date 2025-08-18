import os
import json
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.makedirs("ranks", exist_ok=True)

# === Load preprocessed corpora ===
df_fr = pd.read_csv("preprocessed_data/corpus_fr_clean.csv")
df_nl = pd.read_csv("preprocessed_data/corpus_nl_clean.csv")
corpus_df = pd.concat([df_fr, df_nl], ignore_index=True)

corpus_texts = corpus_df["article"].astype(str).tolist()
corpus_ids = corpus_df["id"].astype(str).tolist()

print(f"Corpus loaded: {len(corpus_texts)} documents.")

vectorizer = TfidfVectorizer(lowercase=True)
corpus_tfidf = vectorizer.fit_transform(corpus_texts)  # shape: [n_docs, vocab_size]

print(f"TF-IDF vectorizer fitted on corpus.")

for lang in ["fr", "nl"]:
    print(f"\nProcessing language: {lang.upper()}")

    df_q = pd.read_csv(f"preprocessed_data/queries_{lang}_clean.csv")

    queries = [
        (str(row["id"]), row["question"], row["article_ids"])
        for _, row in df_q.iterrows()
    ]

    simple_output = {}      # {query_id: [doc_id, …]}
    detailed_output = []    # list of dicts

    for query_id, question, relevant_str in tqdm(queries, desc=f"Ranking queries {lang}"):
        query_tfidf = vectorizer.transform([question])
        similarities = cosine_similarity(query_tfidf, corpus_tfidf).flatten()

        ranked_indices = similarities.argsort()[::-1]
        ranked_doc_ids = [corpus_ids[i] for i in ranked_indices]

        simple_output[query_id] = ranked_doc_ids

        detailed_list = [
            {
                "doc_id": corpus_ids[i],
                "score": float(similarities[i]),
                "rank": rank + 1
            }
            for rank, i in enumerate(ranked_indices)
        ]

        detailed_output.append({
            "query_id": query_id,
            "relevant_ids": [x.strip() for x in relevant_str.split(",")],
            "tfidf_ranked_list": detailed_list
        })

    # === Save outputs ===
    with open(f"ranks/tfidf_ranked_results_{lang}.json", "w", encoding="utf-8") as f:
        json.dump(simple_output, f, ensure_ascii=False, indent=2)

    with open(f"ranks/tfidf_ranked_results_{lang}_with_scores.jsonl", "w", encoding="utf-8") as f:
        for entry in detailed_output:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print("\nTF-IDF ranking completed.")