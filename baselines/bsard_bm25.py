import pandas as pd
import json
import math
import itertools
from statistics import mean
from tqdm import tqdm
import os

class BM25Retriever:
    def __init__(self, corpus_texts, k1=1.0, b=0.6):
        self.corpus = corpus_texts
        self.N = len(corpus_texts)
        self.k1 = k1
        self.b = b
        self.vocab = self._build_vocabulary()
        self.idfs = self._compute_idfs()
        self.avgdl = self._compute_avgdl()
        
    def _build_vocabulary(self):
        return sorted(set(itertools.chain.from_iterable(
            [doc.lower().split() for doc in self.corpus]
        )))
    
    def _compute_idfs(self):
        idfs = dict.fromkeys(self.vocab, 0)
        for word in self.vocab:
            df = sum(1 for doc in self.corpus if word in doc)
            idfs[word] = math.log10((self.N - df + 0.5) / (df + 0.5))
        return idfs
    
    def _compute_avgdl(self):
        return mean(len(doc.split()) for doc in self.corpus)
    
    def _compute_tf(self, term, doc):
        return doc.split().count(term)
    
    def score(self, query, doc):
        score = 0.0
        for t in query.lower().split():
            tf = self._compute_tf(t, doc)
            idf = self.idfs.get(t, math.log10((self.N + 0.5)/0.5))
            denom = tf + self.k1 * (1 - self.b + self.b * len(doc.split())/self.avgdl)
            score += idf * ((tf * (self.k1 + 1)) / denom) if denom > 0 else 0
        return score
    
    def rank(self, query, top_k=None):
        scores = [(i, self.score(query, doc)) for i, doc in enumerate(self.corpus)]
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)
        if top_k:
            ranked = ranked[:top_k]
        return ranked


def run_bm25(language, corpus_path, queries_path, output_dir):
    print(f"\nProcessing language: {language.upper()}")
    
    df_corpus = pd.read_csv(corpus_path)
    corpus_texts = df_corpus["article"].astype(str).tolist()
    corpus_ids = df_corpus["id"].astype(str).tolist()

    df_queries = pd.read_csv(queries_path)

    bm25 = BM25Retriever(corpus_texts, k1=1.0, b=0.6)

    results = {}
    for _, row in tqdm(df_queries.iterrows(), total=len(df_queries), desc=f"Ranking {language} queries"):
        qid = str(row["id"])
        query_text = row["question"]
        ranked = bm25.rank(query_text)
        ranked_doc_ids = [corpus_ids[idx] for idx, _ in ranked]
        results[qid] = ranked_doc_ids

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"bm25_bsard_results_{language}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {out_path}")

os.makedirs("results_bsard", exist_ok=True)

run_bm25(
    language="fr",
    corpus_path="../data_processing/data/original_csv/corpus_fr.csv",
    queries_path="../data_processing/data/original_csv/original_queries_fr.csv",
    output_dir="results_bsard"
)

run_bm25(
    language="nl",
    corpus_path="../data_processing/data/original_csv/corpus_nl.csv",
    queries_path="../data_processing/data/original_csv/original_queries_nl.csv",
    output_dir="results_bsard"
)