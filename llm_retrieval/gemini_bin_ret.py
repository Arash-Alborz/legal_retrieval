import os
import json
import random
import time
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path
from google import genai

# === Setup ===
load_dotenv()
client = genai.Client()

# Choose language
lang = 'nl'  # or 'fr'

# Output folder
output_dir = Path("retrievals")
output_dir.mkdir(parents=True, exist_ok=True)

# Load corpus
corpus_csv_path = f"../data_processing/data/cleaned_corpus/corpus_{lang}_cleaned.csv"
df_corpus = pd.read_csv(corpus_csv_path)
id_to_doc = dict(zip(df_corpus['id'].astype(str), df_corpus['article']))

# Load queries
queries_csv_path = f"../data_processing/data/cleaned_queries_csv/cleaned_test_queries_{lang}.csv"
df_queries = pd.read_csv(queries_csv_path)
query_texts = {str(row['id']): row['question'] for _, row in df_queries.iterrows()}

# Load hard negatives
with open(f"../sampling_hard_negatives/hard_negatives/hard_negatives_{lang}.jsonl", "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]


entries = entries[9:10]  # for debugging

results_jsonl = []

def build_user_message(query_id, query_text, candidate_docs):
    # === JSON skeleton ===
    relevance_dict = {doc['doc_id']: "?" for doc in candidate_docs}
    json_skeleton = {
        "query_id": str(query_id),
        "relevance": relevance_dict
    }

    # === System message ===
    system_msg = (
        "You are an experienced legal assistant in Belgian law, specialized in identifying relevant documents to answer legal questions. "
        "You are precise, concise, and follow the instructions exactly.\n\n"
    )

    # === User task and question ===
    msg = (
        "You are given a legal question and 100 articles. Your task is to assess the relevance of **each article** to answering the question. "
        "You MUST return exactly the JSON object shown below, updating only the values of the 'relevance' field. "
        "For each article ID, replace every ? with a string of:\n"
        "  - 1 if the article is relevant\n"
        "  - 0 if it is NOT relevant\n\n"
        "Do not leave any ?.\n"
        "Do not add, remove, or rename any keys.\n"
        "Do not skip any article.\n"
        f"Question:\n{query_text}\n\n"
        f"Articles:\n"
    )

    for doc in candidate_docs:
        doc_id = doc['doc_id']
        article = id_to_doc[doc_id].strip().replace("\n", " ")
        article = " ".join(article.split())
        msg += f"<{doc_id}>: {article}\n\n"

    msg += (
        "Below is the JSON object for you to edit. "
        "Change only the values of the 'relevance' field as per your judgment and return ONLY the updated JSON:\n\n"
        f"{json.dumps(json_skeleton, indent=2)}"
    )

    return system_msg + msg

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
                "max_output_tokens": 1000,
                }
        )

        result_text = response.text.strip() if response.text else None
        if result_text is None:
            print(f"Empty response for query {query_id}")
            continue

        # parse and re-serialize JSON (clean format, avoid malformed .jsonl)
        try:
            parsed = json.loads(result_text)
            results_jsonl.append(parsed)
        except Exception as e:
            print(f"JSON parsing failed for query {query_id}: {e}")
            print("Raw result:\n", result_text)
            continue

        print(f"\n--- Query ID: {query_id} ---")
        print(f"Question: {query_text}")
        print(f"Gold IDs: {gold_ids}")
        print(f"Gemini Answer:\n{json.dumps(parsed, indent=2)}\n")

        time.sleep(1)

    except Exception as e:
        print(f"Error with query {query_id}: {e}")
        continue

# === Save results ===
all_results_file = output_dir / f"gpt4.1.mini_bin_class_retrievals_{lang}.jsonl"
with open(all_results_file, "w", encoding="utf-8") as f_out:
    for line in results_jsonl:
        f_out.write(line + "\n")

print(f"\nAll queries processed. Results saved to: {output_path}")