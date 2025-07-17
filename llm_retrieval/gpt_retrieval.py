# gpt_retrieval.py

import os
import json
from datasets import load_from_disk
from openai import OpenAI
import random

os.environ["OPENAI_API_KEY"] = "sk-REDACTED"  
client = OpenAI()

corpus = load_from_disk("/Users/arashalborz/Desktop/llm_legal_document_retrieval/data_processing/data/cleaned_corpus_ds/cleaned_corpus")['fr']  # or 'nl'
id_to_doc = {str(doc['id']): doc['article'] for doc in corpus}

with open("/Users/arashalborz/Desktop/llm_legal_document_retrieval/sampling_hard_negatives/data/bm25_sampling/bm25_with_scores_and_ranks_fr.jsonl", "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]


entries = entries[80:85] 

def build_prompt(query_text, candidate_docs):
    prompt = (
        "You are an experienced legal assistant specialized in retrieving relevant documents to answer legal questions.  "
        "You will be given a legal question and 100 articles. Your job is to return the ID of the document that best answers the question.\n\n"
        f"Question:\n{query_text}\n\nDocuments:\n"
    )
    for doc in candidate_docs:
        doc_id = doc['doc_id']
        article = id_to_doc.get(doc_id, "<MISSING>").strip().replace("\n", " ")
        article = article[:500]
        prompt += f"[{doc_id}] {article}\n\n"
    prompt += "Which document is most relevant? Answer with the document ID only."
    return prompt

# looping
for entry in entries:
    query_id = entry['query_id']
    query_text = entry['query_text']
    relevant_ids = entry['relevant_ids']

    # Combine relevant + hard negatives
    candidate_ids = relevant_ids + [doc['doc_id'] for doc in entry['hard_negatives']]
    random.shuffle(candidate_ids)  # Shuffle to avoid position bias

    # Build prompt
    candidate_docs = [{"doc_id": doc_id} for doc_id in candidate_ids]
    prompt = build_prompt(query_text, candidate_docs)

    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # or "gpt-3.5-turbo"
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=10
    )

    answer = response.choices[0].message.content.strip()

    print(f"\n--- Query ID: {query_id} ---")
    print(f"Question: {query_text}")
    print(f"GPT Answer: {answer}")
    print(f"Gold IDs: {relevant_ids}")
