import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# config
lang = "nl"  # or "nl"
CANDIDATES_FILE = f"../sampling_hard_negatives/hard_negatives/hard_negatives_{lang}.jsonl"
DOC_EMB_PATH = f"embeddings_jina/jina_document_embeddings_{lang}.pt"
DOC_IDS_PATH = f"embeddings_jina/jina_document_ids_{lang}.json"
QUERIES_PATH = f"../data_processing/data/cleaned_queries_csv/cleaned_test_queries_{lang}.csv"
OUTPUT_SIMPLE = f"ranks/jina_reranked_{lang}.json"
OUTPUT_DETAILED = f"ranks/jina_reranked_{lang}_with_scores.jsonl"

MODEL_NAME = "jinaai/jina-embeddings-v3"
os.makedirs("ranks", exist_ok=True)

doc_embeddings = torch.load(DOC_EMB_PATH)  # [N, D]
with open(DOC_IDS_PATH, encoding="utf-8") as f:
    doc_ids = json.load(f)

id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

df_queries = pd.read_csv(QUERIES_PATH)
query_texts = dict(zip(df_queries["id"].astype(str), df_queries["question"]))

# model
device = "cpu"
print(f"Running on {device.upper()}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
model.eval()

def embed_query(text, task="retrieval.query", max_length=512):
    with torch.no_grad():
        output = model.encode(
            [text],
            task=task,
            max_length=max_length
        )
        embedding = torch.from_numpy(output[0])
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
    return embedding.unsqueeze(0)  # [1, D]

ranked_results_simple = {}
ranked_results_detailed = []

with open(CANDIDATES_FILE, encoding="utf-8") as f:
    for line in tqdm(f, desc=f"Processing {lang.upper()} queries"):
        obj = json.loads(line)
        qid = obj["query_id"]
        candidate_doc_ids = obj["candidate_docs"]
        relevant_ids = obj["relevant_ids"]

        # embedding the queries
        query_text = query_texts[qid]
        query_emb = embed_query(query_text, task="retrieval.query")  # [1, D]

        indices = [id_to_index[doc_id] for doc_id in candidate_doc_ids]
        candidate_embs = doc_embeddings[indices]  # [100, D]

        scores = (query_emb @ candidate_embs.T).squeeze()  # [100]

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
            "jina_reranked_list": ranked_list
        })

with open(OUTPUT_SIMPLE, "w", encoding="utf-8") as f:
    json.dump(ranked_results_simple, f, ensure_ascii=False, indent=2)

with open(OUTPUT_DETAILED, "w", encoding="utf-8") as f:
    for entry in ranked_results_detailed:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")