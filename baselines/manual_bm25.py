from rank_bm25 import BM25Okapi
import pandas as pd
from tqdm import tqdm
import json
import os

# BM25 parameters from BSARD paper
k1 = 1.5
b = 0.4

df_fr = pd.read_csv("../data_processing/data/original_csv/corpus_fr.csv")
df_nl = pd.read_csv("../data_processing/data/original_csv/corpus_nl.csv")

corpus_df = pd.concat([df_fr, df_nl], ignore_index=True)
corpus_texts = corpus_df["article"].astype(str).tolist()
corpus_ids = corpus_df["id"].astype(str).tolist()
tokenized_corpus = [doc.lower().split() for doc in corpus_texts]

# building BM25 with tuned parameters
bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)

os.makedirs("ranks", exist_ok=True)

for lang, query_file in [
    ("fr", "../data_processing/data/original_csv/original_queries_fr.csv"),
    ("nl", "../data_processing/data/original_csv/original_queries_nl.csv")
]:
    print(f"\nRanking BM25 for language: {lang.upper()}")

    df_queries = pd.read_csv(query_file)
    queries = [
        (str(row["id"]), row["question"])
        for _, row in df_queries.iterrows()
    ]

    ranked_results = {}

    for query_id, question in tqdm(queries, desc=f"Processing {lang.upper()}"):
        query_tokens = question.lower().split()
        scores = bm25.get_scores(query_tokens)
        ranked_indices = scores.argsort()[::-1]  # highest first
        ranked_doc_ids = [corpus_ids[i] for i in ranked_indices]
        ranked_results[query_id] = ranked_doc_ids

    # Save results
    out_path = f"ranks/bm25_ranked_results_{lang}_original.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ranked_results, f, ensure_ascii=False, indent=2)