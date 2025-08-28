import os
import json
import random
import time
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path
from together import Together


load_dotenv()
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

# language
lang = 'nl'  
output_dir = Path("retrievals")
output_dir.mkdir(parents=True, exist_ok=True)

corpus_csv_path = f"../data_processing/data/cleaned_corpus/corpus_{lang}_cleaned.csv"
queries_csv_path = f"../data_processing/data/cleaned_queries_csv/cleaned_test_queries_{lang}.csv"
hard_negatives_path = f"../sampling_hard_negatives/hard_negatives/hard_negatives_{lang}.jsonl"
output_file_path = output_dir / f"qwen3.235b_bin_class_retrieval_{lang}.jsonl"

df_corpus = pd.read_csv(corpus_csv_path)
id_to_doc = dict(zip(df_corpus['id'].astype(str), df_corpus['article']))

df_queries = pd.read_csv(queries_csv_path)
query_texts = {str(row['id']): row['question'] for _, row in df_queries.iterrows()}

with open(hard_negatives_path, "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]

entries=entries[150:]

# prompt builder
def build_user_message(query_id, query_text, candidate_docs):
    relevance_dict = {doc['doc_id']: "?" for doc in candidate_docs}
    json_skeleton = {
        "query_id": str(query_id),
        "relevance": relevance_dict
    }

    msg = (
        "You are given a legal question and 100 articles. Your task is to assess the relevance of **each article** to answering the question. "
        "You MUST return exactly the JSON object shown below, updating only the values of the 'relevance' field. "
        "For each article ID, replace every ? with:\n"
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

    return msg

results_jsonl = []

for entry in tqdm(entries, desc=f"Processing queries for {lang.upper()}"):
    query_id = entry['query_id']
    query_text = query_texts[query_id]
    gold_ids = entry['relevant_ids']

    candidate_ids = entry['candidate_docs']
    random.shuffle(candidate_ids)
    candidate_docs = [{"doc_id": doc_id} for doc_id in candidate_ids]

    user_message = build_user_message(query_id, query_text, candidate_docs)

    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an experienced legal assistant in Belgian law, specialized in identifying relevant documents to answer legal questions. "
                        "You are precise, concise, and follow the instructions exactly."
                    )
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            temperature=0.0,
            max_tokens=1500
        )

        raw_answer = response.choices[0].message.content.strip()
        results_jsonl.append(raw_answer)

        print(f"\n--- Query ID: {query_id} ---")
        print(f"Question: {query_text}")
        print(f"Gold IDs: {gold_ids}")
        print(f"Qwen Answer:\n{raw_answer}\n")

        time.sleep(15)  # sleep time fot limit

    except Exception as e:
        print(f"Error with query {query_id}: {e}")
        continue

with open(output_file_path, "w", encoding="utf-8") as f_out:
    for line in results_jsonl:
        f_out.write(line + "\n")

print(f"\nAll queries processed. Results saved in: {output_file_path}")