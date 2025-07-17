import pandas as pd
import json
import os
from tqdm import tqdm


os.makedirs("gold", exist_ok=True)

for lang, query_file in [
    ("fr", "../data_processing/data/original_csv/original_queries_fr.csv"),
    ("nl", "../data_processing/data/original_csv/original_queries_nl.csv")
]:
    print(f"\nExtracting gold data for language: {lang.upper()}")

    df_queries = pd.read_csv(query_file)

    gold_data = {}

    for _, row in tqdm(df_queries.iterrows(), total=len(df_queries), desc=f"Processing {lang.upper()}"):
        query_id = str(row["id"])
        relevant_str = row["article_ids"]
        relevant_ids = [doc_id.strip() for doc_id in relevant_str.split(",") if doc_id.strip()]
        gold_data[query_id] = relevant_ids

    out_path = f"gold/gold_standard_{lang}_original.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gold_data, f, ensure_ascii=False, indent=2)
