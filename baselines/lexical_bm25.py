from rank_bm25 import BM25Okapi
import pandas as pd
from tqdm import tqdm
import json
import os

# Load preprocessed corpora
df_fr = pd.read_csv("preprocessed_data/corpus_fr_clean.csv")
df_nl = pd.read_csv("preprocessed_data/corpus_nl_clean.csv")

corpus_df = pd.concat([df_fr, df_nl], ignore_index=True)
corpus_texts = corpus_df["article"].astype(str).tolist()
corpus_ids = corpus_df["id"].astype(str).tolist()
tokenized_corpus = [doc.split() for doc in corpus_texts]  # already lowercased & lemmatized

#bm25 = BM25Okapi(tokenized_corpus)
bm25 = BM25Okapi(tokenized_corpus, k1=1.0, b=0.6)

# Load preprocessed queries
df_q_fr = pd.read_csv("preprocessed_data/queries_fr_clean.csv")
df_q_nl = pd.read_csv("preprocessed_data/queries_nl_clean.csv")

queries_by_lang = {
    "fr": [
        (str(row["id"]), row["question"], row["article_ids"])
        for _, row in df_q_fr.iterrows()
    ],
    "nl": [
        (str(row["id"]), row["question"], row["article_ids"])
        for _, row in df_q_nl.iterrows()
    ]
}

os.makedirs("ranks", exist_ok=True)

for lang, queries in queries_by_lang.items():
    print(f"\nProcessing language: {lang.upper()}")

    ranked_results_simple = {}       # {query_id: [doc_ids]}
    ranked_results_with_scores = []  # list of dicts

    for query_id, query_text, relevant_str in tqdm(queries, desc=f"Ranking {lang.upper()} queries"):
        query_tokens = query_text.split()  # already preprocessed

        scores = bm25.get_scores(query_tokens)
        ranked_indices = scores.argsort()[::-1]

        ranked_doc_ids = [corpus_ids[i] for i in ranked_indices]
        ranked_results_simple[query_id] = ranked_doc_ids

        ranked_list = [
            {"doc_id": corpus_ids[i], "score": float(scores[i]), "rank": rank + 1}
            for rank, i in enumerate(ranked_indices)
        ]

        ranked_results_with_scores.append({
            "query_id": query_id,
            "relevant_ids": [x.strip() for x in relevant_str.split(",")],
            "bm25_ranked_list": ranked_list
        })

    with open(f"ranks/bm25_ranked_results_{lang}.json", "w", encoding="utf-8") as f:
        json.dump(ranked_results_simple, f, ensure_ascii=False, indent=2)

    with open(f"ranks/bm25_ranked_results_with_scores_{lang}.jsonl", "w", encoding="utf-8") as f:
        for entry in ranked_results_with_scores:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")