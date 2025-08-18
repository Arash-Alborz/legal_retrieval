import os
import json
import random
import time
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from together import Together
from pathlib import Path

# Load environment variables via env
load_dotenv()
client = Together()

# choose language
lang = 'nl'  # or 'fr'

# output folder
output_dir = Path("retrievals/txt")
output_dir.mkdir(parents=True, exist_ok=True)

# load corpus
corpus_csv_path = f"../data_processing/data/cleaned_corpus/corpus_{lang}_cleaned.csv"
df_corpus = pd.read_csv(corpus_csv_path)
id_to_doc = dict(zip(df_corpus['id'].astype(str), df_corpus['article']))

# load queries
queries_csv_path = f"../data_processing/data/cleaned_queries_csv/cleaned_test_queries_{lang}.csv"
df_queries = pd.read_csv(queries_csv_path)
query_texts = {str(row['id']): row['question'] for _, row in df_queries.iterrows()}

# load hard negatives
with open(
    f"../sampling_hard_negatives/hard_negatives/hard_negatives_{lang}.jsonl",
    "r",
    encoding="utf-8"
) as f:
    entries = [json.loads(line) for line in f]

#entries = entries[39:40]  # test

def build_messages(query_id, query_text, candidate_docs):
    system_message = (
        "You are an experienced legal assistant in Belgian law, specialized in identifying relevant documents to answer legal questions. "
        "You are precise, concise, and follow the instructions exactly."
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
        f"If no documents are relevant, leave the list empty.\n"
        f"Example output:\n"
        f"query id: 4\n"
        f"relevant articles: 5851, 2242\n"
        f"or if none:\n"
        f"query id: 4\n"
        f"relevant articles:\n"
        f"Output only these two lines and nothing else."
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

results_txt = []

for entry in tqdm(entries, desc=f"Processing queries for {lang.upper()}"):
    query_id = entry['query_id']
    query_text = query_texts[query_id]
    gold_ids = entry['relevant_ids']

    candidate_ids = entry['candidate_docs']
    random.shuffle(candidate_ids)

    candidate_docs = [{"doc_id": doc_id} for doc_id in candidate_ids]
    messages = build_messages(query_id, query_text, candidate_docs)

    try:
        response = client.chat.completions.create(                                       
            model="Qwen/Qwen3-235B-A22B-Instruct-2507-tput", 
            messages=messages,
            temperature=0.0,
            max_tokens=200
        )
    except Exception as e:
        print(f"Error with query {query_id}: {e}")
        continue

    choice = response.choices[0]
    raw_answer = choice.message.content.strip()

    results_txt.append(raw_answer + "\n")

    print(f"\n--- Query ID: {query_id} ---")
    print(f"Question: {query_text}")
    print(f"Gold IDs: {gold_ids}")
    print(f"Qwen Answer:\n{raw_answer}\n")

    time.sleep(10)  # sleep time for TPM limit

all_results_file = output_dir / f"qwen3-235B_id_retrieval_{lang}.txt"
with open(all_results_file, "w", encoding="utf-8") as f_out:
    f_out.writelines(results_txt)
