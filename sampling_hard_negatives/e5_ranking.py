import os
import json
import torch
import random
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

lang = "fr" 
CANDIDATES_FILE = f"hard_negatives/hard_negatives_{lang}.jsonl"
DOC_EMB_PATH = f"../baselines/embeddings/document_embeddings_{lang}.pt"
DOC_IDS_PATH = f"../baselines/embeddings/document_ids_{lang}.json"
QUERIES_PATH = f"../data_processing/data/cleaned_queries_csv/cleaned_test_queries_{lang}.csv"
OUTPUT_SIMPLE = f"ranks/e5_reranked_{lang}.json"
OUTPUT_DETAILED = f"ranks/e5_reranked_{lang}_with_scores.jsonl"

MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
os.makedirs("ranks", exist_ok=True)

doc_embeddings = torch.load(DOC_EMB_PATH)
with open(DOC_IDS_PATH) as f:
    doc_ids = json.load(f)

id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

# queries
df_queries = pd.read_csv(QUERIES_PATH)
query_texts = dict(zip(df_queries["id"].astype(str), df_queries["question"]))

# model
device = "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)

def format_query(question):
    task = "Given a legal search query, retrieve relevant documents that answer the query"
    return f"Instruct: {task}\nQuery: {question}"

ranked_results_simple = {}
ranked_results_detailed = []

with open(CANDIDATES_FILE, encoding="utf-8") as f:
    for line in tqdm(f, desc=f"Processing {lang.upper()} queries"):
        obj = json.loads(line)
        qid = obj["query_id"]
        candidate_doc_ids = obj["candidate_docs"]
        relevant_ids = obj["relevant_ids"]

        # embed query
        query_text = format_query(query_texts[qid])
        query_emb = model.encode(query_text, convert_to_tensor=True, normalize_embeddings=True)

        # get candidate document embeddings
        indices = [id_to_index[doc_id] for doc_id in candidate_doc_ids]
        candidate_embs = doc_embeddings[indices]

        # compute scores
        scores = (query_emb @ candidate_embs.T).squeeze()

        # rank within candidate set
        ranked = sorted(
            zip(candidate_doc_ids, scores.tolist()),
            key=lambda x: x[1],
            reverse=True
        )

        ranked_doc_ids = [doc_id for doc_id, _ in ranked]
        ranked_results_simple[qid] = ranked_doc_ids

        ranked_list = [
            {"doc_id": doc_id, "score": float(score), "rank": i+1}
            for i, (doc_id, score) in enumerate(ranked)
        ]

        ranked_results_detailed.append({
            "query_id": qid,
            "relevant_ids": relevant_ids,
            "e5_reranked_list": ranked_list
        })

with open(OUTPUT_SIMPLE, "w", encoding="utf-8") as f:
    json.dump(ranked_results_simple, f, ensure_ascii=False, indent=2)

with open(OUTPUT_DETAILED, "w", encoding="utf-8") as f:
    for entry in ranked_results_detailed:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")