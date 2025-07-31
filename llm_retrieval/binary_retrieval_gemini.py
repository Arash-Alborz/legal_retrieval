from google import genai
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import json
import os
import random
from pydantic import BaseModel, Field
from typing import Dict
import time

# === Setup ===
load_dotenv()
client = genai.Client()

#class BinaryRelevanceResult(BaseModel):
#    query_id: str
#    relevance: Dict[str, str]  # article_id -> "0" or "1" as string

lang = 'nl'
output_dir = Path("retrievals")
output_dir.mkdir(parents=True, exist_ok=True)

corpus_csv_path = f"../data_processing/data/cleaned_corpus/corpus_{lang}_cleaned.csv"
df_corpus = pd.read_csv(corpus_csv_path)
id_to_doc = dict(zip(df_corpus['id'].astype(str), df_corpus['article']))

queries_csv_path = f"../data_processing/data/cleaned_queries_csv/cleaned_test_queries_{lang}.csv"
df_queries = pd.read_csv(queries_csv_path)
query_texts = {str(row['id']): row['question'] for _, row in df_queries.iterrows()}

with open(f"../sampling_hard_negatives/hard_negatives/hard_negatives_{lang}.jsonl", "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]

entries = entries[:1]  # for debugging

results_jsonl = []

def build_user_message(query_id, query_text, candidate_docs):
    system_message = (
        "You are an experienced legal assistant in Belgian law, specialized in identifying relevant documents to answer legal questions. "
        "You are precise, concise, and follow the instructions exactly.\n\n"
    )

    user_message = (
        "You are given a legal question and 100 articles. Your task is to assess the relevance of **each article** to answering the question. "
        "You MUST return exactly the JSON object described below. Use the query ID as the top-level key. "
        "For each article ID, assign a binary relevance value:\n"
        "  - \"1\" if the article is relevant\n"
        "  - \"0\" if the article is NOT relevant\n\n"
        "Do NOT leave any articles unscored.\n"
        "Do NOT add, remove, or rename any keys.\n"
        "Do NOT hallucinate new IDs.\n\n"
        "Question:\n"
        f"{query_text}\n\n"
        "Articles:\n"
    )

    for doc in candidate_docs:
        doc_id = doc['doc_id']
        article = id_to_doc[doc_id].strip().replace("\n", " ")
        article = " ".join(article.split())
        user_message += f"[{doc_id}] {article}\n\n"

    user_message += (
        f"Return only the following JSON object, updating the relevance values appropriately:\n\n"
        f"{{\n"
        f"  \"{query_id}\": {{\n"
        f"    \"<doc_id1>\": \"0\",\n"
        f"    \"<doc_id2>\": \"1\",\n"
        f"    ...\n"
        f"  }}\n"
        f"}}"
    )

    return system_message + user_message

# === Main loop ===
for entry in tqdm(entries, desc=f"Processing queries for {lang.upper()}"):
    query_id = entry['query_id']
    query_text = query_texts[query_id]
    gold_ids = entry['relevant_ids']

    candidate_ids = entry['candidate_docs']
    random.shuffle(candidate_ids)
    candidate_docs = [{"doc_id": doc_id} for doc_id in candidate_ids]

    user_message = build_user_message(query_id, query_text, candidate_docs)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=user_message,
            config={
                #"response_mime_type": "application/json",
                "thinking_config": {"thinking_budget": 0},
                "temperature": 0.0,
                #"max_output_tokens": 5000,
            }
        )

        result_text = response.text.strip()
        result_obj = json.loads(result_text)
        results_jsonl.append(result_obj)

        print(f"\n--- Query ID: {query_id} ---")
        print(f"Question: {query_text}")
        print(f"Gold IDs: {gold_ids}")
        print(f"Gemini Answer:\n{json.dumps(result_obj, indent=2)}\n")

        time.sleep(10)

    except Exception as e:
        print(f"Error with query {query_id}: {e}")
        continue

# === Save ===
output_path = output_dir / f"gemini_2.5.flash.lite_bin__class_retrieval_{lang}.jsonl"
with open(output_path, "w", encoding="utf-8") as fout:
    for obj in results_jsonl:
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"\nAll queries processed. Results saved to: {output_path}")