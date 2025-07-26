import os
import json
import random
from tqdm import tqdm

OUTPUT_DIR = "hard_negatives"
TOP_K = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

for lang in ["nl", "fr"]:
    bm25_file = os.path.join(f"../baselines/ranks/bm25_ranked_results_{lang}.json")
    gold_file = os.path.join(f"gold_standard_{lang}.json")
    output_file = os.path.join(OUTPUT_DIR, f"hard_negatives_{lang}.jsonl")

    print(f"\nProcessing language: {lang.upper()}")

    with open(bm25_file, encoding="utf-8") as f:
        bm25_ranks = json.load(f)

    with open(gold_file, encoding="utf-8") as f:
        gold_data = json.load(f)

    with open(output_file, "w", encoding="utf-8") as out_f:
        for qid in tqdm(sorted(bm25_ranks.keys(), key=int), desc=f"Building candidates {lang}"):
            bm25_top = bm25_ranks[qid][:TOP_K]
            relevant = set(gold_data[qid])

            # find which relevant documents are missing
            missing_relevant = [doc for doc in relevant if doc not in bm25_top]

            # replace missing relevant docs --> dropping last ranks
            if missing_relevant:
                num_to_replace = len(missing_relevant)
                bm25_top = bm25_top[:-num_to_replace] + missing_relevant

            # sanity check
            assert all(doc in bm25_top for doc in relevant), f"Query {qid}: relevant docs missing after adjustment."

            # shuffle
            random.shuffle(bm25_top)

            out_obj = {
                "query_id": qid,
                "candidate_docs": bm25_top,
                "relevant_ids": list(relevant)
            }
            out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(f"Saved candidate sets to: {output_file}")