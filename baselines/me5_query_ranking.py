import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# config
LANG = "fr"   # change to "fr" for French
QUERY_PATH = f"../data_processing/data/cleaned_queries_csv/cleaned_test_queries_{LANG}.csv"
DOC_EMB_PATH = f"embeddings/document_embeddings_{LANG}.pt"
DOC_IDS_PATH = f"embeddings/document_ids_{LANG}.json"
OUTPUT_SIMPLE = f"ranks/e5_ranked_results_{LANG}.json"
OUTPUT_DETAILED = f"ranks/e5_ranked_results_{LANG}_with_scores.jsonl"

MODEL_NAME = "intfloat/multilingual-e5-large-instruct"

os.makedirs("ranks", exist_ok=True)

def format_query(question):
    task = "Given a legal search query, retrieve relevant documents that answer the query"
    return f"Instruct: {task}\nQuery: {question}"

doc_embeddings = torch.load(DOC_EMB_PATH)
with open(DOC_IDS_PATH) as f:
    doc_ids = json.load(f)

print(f"Loaded {len(doc_ids)} documents & embeddings.")

df_queries = pd.read_csv(QUERY_PATH)
queries = df_queries[["id", "question", "article_ids"]].astype(str).values.tolist()

print(f"Loaded {len(queries)} queries.")

device = "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)

ranked_results_simple = {}
ranked_results_detailed = []

for qid, question, relevant_str in tqdm(queries, desc="Processing queries"):
    query_text = format_query(question)

    query_emb = model.encode(query_text, convert_to_tensor=True, normalize_embeddings=True)
    scores = (query_emb @ doc_embeddings.T).squeeze()

    ranked_indices = scores.argsort(descending=True)
    ranked_doc_ids = [doc_ids[i] for i in ranked_indices.tolist()]
    ranked_results_simple[qid] = ranked_doc_ids

    ranked_list = [
        {
            "doc_id": doc_ids[i],
            "score": float(scores[i]),
            "rank": rank + 1
        }
        for rank, i in enumerate(ranked_indices.tolist())
    ]

    ranked_results_detailed.append({
        "query_id": qid,
        "relevant_ids": [x.strip() for x in relevant_str.split(",")],
        "e5_ranked_list": ranked_list
    })

with open(OUTPUT_SIMPLE, "w", encoding="utf-8") as f:
    json.dump(ranked_results_simple, f, ensure_ascii=False, indent=2)

with open(OUTPUT_DETAILED, "w", encoding="utf-8") as f:
    for entry in ranked_results_detailed:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
