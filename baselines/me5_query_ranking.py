import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ===== Config =====
LANG = "nl"  # or "nl"
QUERY_PATH = f"../data_processing/data/cleaned_queries_csv/cleaned_test_queries_{LANG}.csv"

DOC_EMB_PATH = f"embeddings/document_embeddings_{LANG}.pt"
DOC_IDS_PATH = f"embeddings/document_ids_{LANG}.json"

OUTPUT_JSONL = f"ranks/e5_top100_ranks_{LANG}.jsonl"
TOP_K = 100

# Set this to True to rank ONLY within the 100 candidates from hard_negatives (recommended for fair comparison)
USE_CANDIDATES = True
CANDIDATES_JSONL = f"../sampling_hard_negatives/hard_negatives/hard_negatives_{LANG}.jsonl"

MODEL_NAME = "intfloat/multilingual-e5-large-instruct"

os.makedirs("ranks", exist_ok=True)

def format_query(question: str) -> str:
    task = "Given a legal search query, retrieve relevant documents that answer the query"
    return f"Instruct: {task}\nQuery: {question}"

# ===== Load corpus embeddings & IDs =====
doc_embeddings: torch.Tensor = torch.load(DOC_EMB_PATH)  # shape: [N, d], assumed L2-normalized
with open(DOC_IDS_PATH, "r", encoding="utf-8") as f:
    doc_ids = [str(x) for x in json.load(f)]

id2idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
N_docs = len(doc_ids)
print(f"Loaded {N_docs} documents & embeddings.")

# ===== Load queries =====
df_queries = pd.read_csv(QUERY_PATH)
queries = df_queries[["id", "question"]].astype(str).values.tolist()
print(f"Loaded {len(queries)} queries.")

# ===== Optionally load 100-candidate sets =====
query_to_candidates = None
if USE_CANDIDATES:
    query_to_candidates = {}
    with open(CANDIDATES_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qid = str(obj["query_id"])
            # Ensure all candidate IDs are strings
            cand_ids = [str(x) for x in obj["candidate_docs"]]
            query_to_candidates[qid] = cand_ids
    print(f"Loaded candidate sets for {len(query_to_candidates)} queries.")

# ===== Model =====
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)

# ===== Rank & write JSONL =====
written = 0
with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
    for qid, question in tqdm(queries, desc="Ranking with mE5"):
        qtext = format_query(question)
        q_emb = model.encode(qtext, convert_to_tensor=True, normalize_embeddings=True)  # [d]
        # similarity = dot product because embeddings are normalized
        if USE_CANDIDATES:
            cand_ids = query_to_candidates.get(qid, [])
            if not cand_ids:
                # Fallback: rank against all documents if no candidates available
                scores = q_emb @ doc_embeddings.T  # [N]
                k = min(TOP_K, N_docs)
                top_scores, top_idx = torch.topk(scores, k=k, largest=True)
                top_ids = [doc_ids[i] for i in top_idx.tolist()]
            else:
                cand_idx = [id2idx[c] for c in cand_ids if c in id2idx]
                cand_mat = doc_embeddings[cand_idx]  # [C, d]
                scores = q_emb @ cand_mat.T          # [C]
                k = min(TOP_K, len(cand_idx))
                top_scores, top_local_idx = torch.topk(scores, k=k, largest=True)
                top_ids = [cand_ids[i] for i in top_local_idx.tolist()]
        else:
            scores = q_emb @ doc_embeddings.T  # [N]
            k = min(TOP_K, N_docs)
            top_scores, top_idx = torch.topk(scores, k=k, largest=True)
            top_ids = [doc_ids[i] for i in top_idx.tolist()]

        # Write eval-ready line
        fout.write(json.dumps({"query_id": str(qid), "ranks": [str(x) for x in top_ids]}, ensure_ascii=False) + "\n")
        written += 1

print(f"Wrote Top-{TOP_K} JSONL for {written} queries to {OUTPUT_JSONL}")