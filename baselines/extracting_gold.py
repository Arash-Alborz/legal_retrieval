import pandas as pd
import json
import os

os.makedirs("gold", exist_ok=True)

languages = ["fr", "nl"]
input_folder = "../data_processing/data/cleaned_queries_csv"


for lang in languages:
    df = pd.read_csv(f"{input_folder}/cleaned_test_queries_{lang}.csv")

    gold = {}
    for _, row in df.iterrows():
        qid = str(row["id"])
        rel_ids = [x.strip() for x in str(row["article_ids"]).split(",") if x.strip()]
        gold[qid] = rel_ids

    output_path = os.path.join(f"gold/gold_standard_{lang}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gold, f, ensure_ascii=False, indent=2)