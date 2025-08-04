import os
import json
import random
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path
import time
from google import genai

# === Setup ===
load_dotenv()
client = genai.Client()

# -------- CONFIG --------
lang = 'nl'  # or 'fr'
output_dir = Path("retrievals/txt")
output_dir.mkdir(parents=True, exist_ok=True)

corpus_csv_path = f"../data_processing/data/cleaned_corpus/corpus_{lang}_cleaned.csv"
queries_csv_path = f"../data_processing/data/cleaned_queries_csv/cleaned_test_queries_{lang}.csv"
hard_negatives_path = f"../sampling_hard_negatives/hard_negatives/hard_negatives_{lang}.jsonl"
output_file_path = output_dir / f"gemini2.0.flash_sorted_ranking_{lang}.txt"

# -------- LOAD DATA --------
df_corpus = pd.read_csv(corpus_csv_path)
id_to_doc = dict(zip(df_corpus['id'].astype(str), df_corpus['article']))

df_queries = pd.read_csv(queries_csv_path)
query_texts = {str(row['id']): row['question'] for _, row in df_queries.iterrows()}

with open(hard_negatives_path, "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]

#entries = entries[:1]  # Limit to first 5 queries

# -------- PROMPT GENERATION --------
def build_messages(query_id, query_text, candidate_docs):
    system_message = (
        "You are an experienced legal assistant in Belgian law, specialized in identifying relevant documents to answer legal questions. "
        "You are precise, concise, and follow the instructions exactly."
    )

    candidate_ids = [doc["doc_id"] for doc in candidate_docs]
    id_list_str = ", ".join(candidate_ids)

    user_message = (
        f"Given the following legal question and 100 articles, rank the articles by how relevant they are to answering the question. "
        "You must sort them from most relevant to least relevant.\n\n"
        "You must include all of the 100 article IDs, even if they are not relevant.\n\n"
        "Do not repeat any IDs. Do not invent any new IDs.\n\n"
        f"Question:\n{query_text}\n\nDocuments:\n"
    )

    for doc in candidate_docs:
        doc_id = doc['doc_id']
        article = id_to_doc[doc_id].strip().replace("\n", " ")
        article = " ".join(article.split())
        user_message += f"<{doc_id}>: {article}\n\n"

    user_message += (
        f"Output only the ranked list of document IDs, sorted from most to least relevant. "
        f"Write exactly two lines. On the first line write: query id: {query_id}\n"
        f"On the second line write: ranked articles: followed by the comma-separated list of article IDs, in order of relevance only.\n\n.\n"
        f"Example output:\nquery id: 4\nranked articles: 5851, 2242, 1950, 1004\n"
        f"Output only these two lines and nothing else.\n"
        f"You must use all of the 100 article IDs in this list:\n[{id_list_str}]\n\n for ranking and only these 100 article IDs and nothing else."
    )

    return system_message + "\n\n" + user_message

# -------- MAIN EXECUTION --------
results_txt = []

for entry in tqdm(entries, desc=f"Processing queries for {lang.upper()}"):
    query_id = entry['query_id']
    query_text = query_texts[query_id]
    candidate_ids = entry['candidate_docs']
    random.shuffle(candidate_ids)

    candidate_docs = [{"doc_id": doc_id} for doc_id in candidate_ids]
    prompt = build_messages(query_id, query_text, candidate_docs)

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "thinking_config": {"thinking_budget": 0},
                "max_output_tokens": 1000
            }
        )

        result_text = response.text.strip()
        result_obj = json.loads(result_text)

    except Exception as e:
        print(f"Error with query {query_id}: {e}")
        continue

    results_txt.append(result_text + "\n")

    print(f"\n--- Query ID: {query_id} ---")
    print(f"Question: {query_text}")
    print(f"Gemini Answer:\n{result_text}")
    time.sleep(30)

# Write all results
with open(output_file_path, "w", encoding="utf-8") as f_out:
    f_out.writelines(results_txt)

print(f"\nAll queries processed. Results saved in: {output_file_path}")