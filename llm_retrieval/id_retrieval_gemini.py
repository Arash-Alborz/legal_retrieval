from google import genai
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import json
import os
import random
from pydantic import BaseModel
from typing import List
import time

# === Setup ===
load_dotenv()
client = genai.Client()

class RetrievalResult(BaseModel):
    query_id: str
    relevant_document_ids: List[str]

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

entries = entries[54:55]  # for testing

results_dict = {}

def build_user_message(query_id, query_text, candidate_docs):
    system_message = (
        "You are an experienced legal assistant in Belgian law, specialized in identifying relevant documents to answer legal questions. "
        "You are precise, concise, and follow the instructions exactly.\n\n"
    )
    user_message = (
        "Given the following legal question and 100 articles, identify which articles are relevant to answering the question. "
        "There may be zero, one, or multiple relevant documents.\n\n"
        f"Question:\n{query_text}\n\nDocuments:\n"
    )
    for doc in candidate_docs:
        doc_id = doc['doc_id']
        article = id_to_doc[doc_id].strip().replace("\n", " ")
        article = " ".join(article.split())
        user_message += f"<{doc_id}>: {article}\n\n"

    user_message += (
        f"You must only select relevant article IDs from the documents listed above. "
        f"Use the IDs exactly as shown inside brackets in front of the article text.\n\n"
        f"Output the result in plain text. Write exactly two lines.\n"
        f"On the first line write: query id: {query_id}\n"
        f"On the second line write: relevant articles: followed by a comma-separated list of the IDs of the relevant documents.\n"
        f"Only list truly relevant article IDs. If you are unsure about an article, do not include it.\n" 
        f"If no documents are relevant, leave the list empty.\n"
        f"Example output:\n"
        f"query id: 4\n"
        f"relevant articles: 5851, 2242\n"
        f"or if none:\n"
        f"query id: 4\n"
        f"relevant articles:\n"
        "Output only these two lines and nothing else."
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
                "response_mime_type": "application/json",
                "response_schema": RetrievalResult,
                "thinking_config": {"thinking_budget": 0}
            }
        )

        result_text = response.text.strip()
        result_obj = json.loads(result_text)

        # Save to dict
        results_dict[result_obj["query_id"]] = result_obj["relevant_document_ids"]

        # Terminal log like GPT
        print(f"\n--- Query ID: {query_id} ---")
        print(f"Question: {query_text}")
        print(f"Gold IDs: {gold_ids}")
        print(f"Gemini Answer:")
        print(f"{result_text}\n")

        time.sleep(20)  #adjust as needed

    except Exception as e:
        print(f"Error with query {query_id}: {e}")
        continue

# === Save output ===
output_path = output_dir / f"gemini_2.5.flash_lite_id_retrieval_{lang}.json"
with open(output_path, "w", encoding="utf-8") as fout:
    json.dump(results_dict, fout, ensure_ascii=False, indent=2)

print(f"\nAll queries processed. Results saved in: {output_path}")