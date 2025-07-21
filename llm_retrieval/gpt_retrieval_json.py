import os
import json
import random
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path

# load .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# choose language
lang = 'nl'  # or 'fr'

# output folder
output_dir = Path("llm_retrieval/results")
output_dir.mkdir(parents=True, exist_ok=True)

# load corpus
corpus_csv_path = f"../data_processing/data/original_csv/corpus_{lang}.csv"
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

# optional: slice for testing
entries = entries[4:5]  # adjust as needed

def build_prompt(query_id, query_text, candidate_docs):
    prompt = (
        "You are an experienced legal assistant specialized in retrieving relevant documents to answer legal questions. "
        "You will be given a legal question and 100 articles. Your job is to decide which documents are relevant to the question. "
        "There may be zero, one, or multiple relevant documents.\n\n"
        f"Question:\n{query_text}\n\nDocuments:\n"
    )
    for doc in candidate_docs:
        doc_id = doc['doc_id']
        article = id_to_doc[doc_id].strip().replace("\n", " ")
        article = " ".join(article.split()[:470])  # truncate to first 500 words
        prompt += f"[{doc_id}] {article}\n\n"
    prompt += (
        f"Output a JSON object with two fields: `query_id` and `relevant_document_ids`, "
        f"where `query_id` is \"{query_id}\" and `relevant_document_ids` is a list of the IDs of relevant documents. "
        f"If no documents are relevant, output an empty list. "
        f"Example:\n"
        f"{{ \"query_id\": \"4\", \"relevant_document_ids\": [\"2222\", \"2242\"] }}\n"
        "Output only valid JSON. Do not include any explanation."
    )
    return prompt

def clean_json_string(s):
    s = s.strip()
    if s.startswith("```json"):
        s = s[len("```json"):].strip()
    if s.startswith("```"):
        s = s[len("```"):].strip()
    if s.endswith("```"):
        s = s[:-3].strip()
    return s

results = []

for entry in tqdm(entries, desc=f"Processing queries for {lang.upper()}"):
    query_id = entry['query_id']
    query_text = query_texts[query_id]
    gold_ids = entry['relevant_ids']

    candidate_ids = entry['candidate_docs']
    random.shuffle(candidate_ids)

    candidate_docs = [{"doc_id": doc_id} for doc_id in candidate_ids]
    prompt = build_prompt(query_id, query_text, candidate_docs)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=200
        )
    except Exception as e:
        print(f"Error with query {query_id}: {e}")
        continue

    choice = response.choices[0]
    raw_answer = choice.message.content.strip()
    usage = response.usage

    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens

    # clean & parse GPT output
    raw_answer_clean = clean_json_string(raw_answer)

    try:
        answer_json = json.loads(raw_answer_clean)
        relevant_docs = answer_json.get("relevant_document_ids", [])
    except json.JSONDecodeError:
        print(f"Failed to parse JSON for query {query_id}: {raw_answer}")
        relevant_docs = []

    # save minimal result
    result = {
        "query_id": query_id,
        "relevant_document_ids": relevant_docs
    }
    results.append(result)

    # log to terminal
    print(f"\n--- Query ID: {query_id} ---")
    print(f"Question: {query_text}")
    print(f"Gold IDs: {gold_ids}")
    print(f"GPT Answer (raw): {raw_answer}")
    print(f"GPT Relevant IDs (parsed): {relevant_docs}")
    print(f"Tokens - Input: {input_tokens}, Output: {output_tokens}\n")

    time.sleep(30)  # delay to avoid TPM limit

# save all results to one file
all_results_file = output_dir / f"results_{lang}.json"
with open(all_results_file, "w", encoding="utf-8") as f_out:
    json.dump(results, f_out, indent=2, ensure_ascii=False)

print(f"\nAll queries processed. Results saved in: {all_results_file}")