import os
import json
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi

# -------- CONFIG --------
LANG = "fr"  # or "nl"
QUERY_PATH = f"preprocessed_data/queries_{LANG}_clean.csv"
CORPUS_PATH = f"preprocessed_data/corpus_{LANG}_clean.csv"
OUTPUT_SIMPLE = f"ranks/bm25_ranked_results_{LANG}.json"
OUTPUT_DETAILED = f"ranks/bm25_ranked_results_{LANG}_with_scores.jsonl"

# ------------------------

os.makedirs("ranks", exist_ok=True)

# Load corpus
df_corpus = pd.read_csv(CORPUS_PATH)
corpus_texts = df_corpus["article"].astype(str).tolist()
corpus_ids = df_corpus["id"].astype(str).tolist()
tokenized_corpus = [doc.split() for doc in corpus_texts]

bm25 = BM25Okapi(tokenized_corpus, k1=1.0, b=0.6)

print(f"Loaded {len(corpus_ids)} documents.")

# Load queries
df_queries = pd.read_csv(QUERY_PATH)
queries = df_queries[["id", "question", "article_ids"]].astype(str).values.tolist()

print(f"Loaded {len(queries)} queries.")

ranked_results_simple = {}
ranked_results_detailed = []

for qid, question, relevant_str in tqdm(queries, desc="Processing queries"):
    query_tokens = question.split()  # already preprocessed

    scores = bm25.get_scores(query_tokens)
    ranked_indices = scores.argsort()[::-1]

    ranked_doc_ids = [corpus_ids[i] for i in ranked_indices]
    ranked_results_simple[qid] = ranked_doc_ids

    ranked_list = [
        {
            "doc_id": corpus_ids[i],
            "score": float(scores[i]),
            "rank": rank + 1
        }
        for rank, i in enumerate(ranked_indices)
    ]

    ranked_results_detailed.append({
        "query_id": qid,
        "relevant_ids": [x.strip() for x in relevant_str.split(",")],
        "bm25_ranked_list": ranked_list
    })

# Write output
with open(OUTPUT_SIMPLE, "w", encoding="utf-8") as f:
    json.dump(ranked_results_simple, f, ensure_ascii=False, indent=2)

with open(OUTPUT_DETAILED, "w", encoding="utf-8") as f:
    for entry in ranked_results_detailed:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")