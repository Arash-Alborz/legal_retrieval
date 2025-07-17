from rank_bm25 import BM25Okapi
import pandas as pd
from tqdm import tqdm
import json
import os

# loading both corpora
df_fr = pd.read_csv("../data_processing/data/original_csv/corpus_fr.csv")
df_nl = pd.read_csv("../data_processing/data/original_csv/corpus_nl.csv")

corpus_df = pd.concat([df_fr, df_nl], ignore_index=True)
corpus_texts = corpus_df["article"].astype(str).tolist()
corpus_ids = corpus_df["id"].astype(str).tolist()
tokenized_corpus = [doc.lower().split() for doc in corpus_texts]

bm25 = BM25Okapi(tokenized_corpus)

df_q_fr = pd.read_csv("../data_processing/data/cleaned_queries_csv/cleaned_test_queries_fr.csv")
df_q_nl = pd.read_csv("../data_processing/data/cleaned_queries_csv/cleaned_test_queries_nl.csv")

queries_by_lang = {
    "fr": [
        (str(row["id"]), row["article_ids"]) for _, row in df_q_fr.iterrows()
    ],
    "nl": [
        (str(row["id"]), row["article_ids"]) for _, row in df_q_nl.iterrows()
    ]
}

os.makedirs("ranks", exist_ok=True)

for lang, queries in queries_by_lang.items():
    print(f"\n🔷 Processing language: {lang.upper()}")

    ranked_results_simple = {}      # {query_id: [doc_ids]}
    ranked_results_with_scores = []  # list of dicts

    for query_id, relevant_str in tqdm(queries, desc=f"Ranking {lang.upper()} queries"):
        query_tokens = query_id.lower().split()
        if lang == "fr":
            question = df_q_fr.loc[df_q_fr["id"].astype(str) == query_id, "question"].values[0]
        else:
            question = df_q_nl.loc[df_q_nl["id"].astype(str) == query_id, "question"].values[0]
        query_tokens = question.lower().split()

        scores = bm25.get_scores(query_tokens)
        ranked_indices = scores.argsort()[::-1]

        ranked_doc_ids = [corpus_ids[i] for i in ranked_indices]
        ranked_results_simple[query_id] = ranked_doc_ids

        ranked_list = [
            {"doc_id": corpus_ids[i], "score": float(scores[i]), "rank": rank+1}
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