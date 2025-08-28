import os
import json
import random
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path
import time
from together import Together

load_dotenv()
client = Together()

lang = 'nl'  # or 'fr'
output_dir = Path("rankings")
output_dir.mkdir(parents=True, exist_ok=True)

corpus_csv_path = f"../data_processing/data/cleaned_corpus/corpus_{lang}_cleaned.csv"
queries_csv_path = f"../data_processing/data/cleaned_queries_csv/cleaned_test_queries_{lang}.csv"
hard_negatives_path = f"../sampling_hard_negatives/hard_negatives/hard_negatives_{lang}.jsonl"
output_file_path = output_dir / f"llama3.3_70b_sorted_ranking_{lang}.txt"

df_corpus = pd.read_csv(corpus_csv_path)
id_to_doc = dict(zip(df_corpus['id'].astype(str), df_corpus['article']))

df_queries = pd.read_csv(queries_csv_path)
query_texts = {str(row['id']): row['question'] for _, row in df_queries.iterrows()}

with open(hard_negatives_path, "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]


#entries= entries[:1] # test
def build_prompt(query_id, query_text, candidate_docs):
    system_message = (
        "You are an experienced legal assistant in Belgian law, specialized in identifying relevant documents to answer legal questions. "
        "You are precise, concise, and follow the instructions exactly."
    )

    candidate_ids = [doc["doc_id"] for doc in candidate_docs]
    id_list_str = ", ".join(candidate_ids)

    user_message = (
        f"Given the following legal question and 100 articles, rank the articles by how relevant they are to answering the question. "
        "You must sort them from most relevant to least relevant.\n\n"
        "You must include all of the 100 article IDs, even if they are not relevant.\n"
        "Do not repeat any IDs. Do not invent any new IDs.\n\n"
        f"Question:\n{query_text}\n\nDocuments:\n"
    )

    for doc in candidate_docs:
        doc_id = doc['doc_id']
        article = id_to_doc[doc_id].strip().replace("\n", " ")
        article = " ".join(article.split())
        user_message += f"<{doc_id}>: {article}\n\n"

    user_message += (
        f"Output the result strictly in JSON format with two keys: 'query_id' and 'ranked_articles'. "
        f"'query_id' must be exactly \"{query_id}\". "
        f"'ranked_articles' is a single string of the 100 article IDs separated by commas, in ranked order. "
        f"Ensure that exactly these 100 article IDs appear, each once, and in your ranked order: [{id_list_str}]\n\n"
        f"Example:\n"
        f"{{\n  \"query_id\": \"{query_id}\",\n  \"ranked_articles\": \"5851, 2242, 1950, 1004, ...\"\n}}\n"
        f"Output only valid JSON and nothing else."
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

results_txt = []

for entry in tqdm(entries, desc=f"Processing queries for {lang.upper()}"):
    query_id = entry['query_id']
    query_text = query_texts[query_id]
    candidate_ids = entry['candidate_docs']
    random.shuffle(candidate_ids)

    candidate_docs = [{"doc_id": doc_id} for doc_id in candidate_ids]
    messages = build_prompt(query_id, query_text, candidate_docs)

    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=messages,
            temperature=0.0,
            max_tokens=700
        )
        choice = response.choices[0]
        result_text = choice.message.content.strip()

        if result_text.startswith("```"):
            result_text = result_text.strip("`").strip()
            if result_text.lower().startswith("json"):
                result_text = result_text[4:].strip()

        try:
            _ = json.loads(result_text)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON for query {query_id}")

    except Exception as e:
        print(f"Error with query {query_id}: {e}")
        continue

    results_txt.append(result_text + "\n")

    print(f"\n--- Query ID: {query_id} ---")
    print(f"Question: {query_text}")
    print(f"LLaMA Answer:\n{result_text}")
    time.sleep(10)

with open(output_file_path, "w", encoding="utf-8") as f_out:
    f_out.writelines(results_txt)