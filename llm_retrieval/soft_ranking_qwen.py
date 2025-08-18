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
client = Together()

lang = 'nl'  # or 'fr'
output_dir = Path("rankings/scored")
output_dir.mkdir(parents=True, exist_ok=True)

# corpus
corpus_csv_path = f"../data/cleaned_corpus/corpus_{lang}_cleaned.csv"
df_corpus = pd.read_csv(corpus_csv_path)
id_to_doc = dict(zip(df_corpus['id'].astype(str), df_corpus['article']))

# queries
queries_csv_path = f"../data/cleaned_queries_csv/cleaned_test_queries_{lang}.csv"
df_queries = pd.read_csv(queries_csv_path)
query_texts = {str(row['id']): row['question'] for _, row in df_queries.iterrows()}

# hard negatives
with open(f"../data/hard_negatives_{lang}.jsonl", "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]

entries = entries[8:17]  

def build_user_message(query_id, query_text, candidate_docs):
    relevance_dict = {doc['doc_id']: "?" for doc in candidate_docs}
    json_skeleton = {
        "query_id": str(query_id),
        "relevance_scores": relevance_dict
    }

    msg = (
        "You are given a legal question and 100 articles. Your task is to assign a relevance **score** to each article. "
        "Scores must be between 1 and 100, where:\n"
        "- 100 = perfectly relevant\n"
        "- 1 = completely irrelevant\n"
        "- Use the full range to reflect subtle differences.\n\n"
        "Do not add, remove, or skip any article IDs.\n"
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
        "Replace every ? with a score between 1 and 100, using only integers. "
        "Return ONLY the completed JSON object, nothing else:\n\n"
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

    messages = [
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
    ]

    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
            messages=messages,
            temperature=0.0,
            max_tokens=1500
        )
        choice = response.choices[0]
        raw_answer = choice.message.content.strip()

        if raw_answer.startswith("```"):
            raw_answer = raw_answer.strip("`").strip()
            if raw_answer.startswith("json"):
                raw_answer = raw_answer[4:].strip()

        results_jsonl.append(raw_answer)

        print(f"\n--- Query ID: {query_id} ---")
        print(f"Question: {query_text}")
        print(f"Gold IDs: {gold_ids}")
        print(f"Qwen Answer:\n{raw_answer}\n")

        time.sleep(1) 

    except Exception as e:
        print(f"Error with query {query_id}: {e}")
        continue

output_file = output_dir / f"qwen3.235b_score_ranking_{lang}.jsonl"
with open(output_file, "w", encoding="utf-8") as f_out:
    for line in results_jsonl:
        f_out.write(line + "\n")