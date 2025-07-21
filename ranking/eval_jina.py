import json
from statistics import mean
from tqdm import tqdm
import os
from math import log2

metrics_at_k = {
    'recall': [10, 20, 50, 100],
    'map': [1, 5, 10, 100],
    'mrr': [1, 5, 10, 100],
    'ndcg': [1, 5, 10, 100]
}

os.makedirs("results", exist_ok=True)


class Evaluator:
    def __init__(self, metrics_at_k):
        self.metrics_at_k = metrics_at_k

    def compute_all_metrics(self, all_results, all_ground_truths):
        scores = dict()
        for k in self.metrics_at_k['recall']:
            scores[f'recall@{k}'] = self.compute_mean(self.recall, all_ground_truths, all_results, k)
        for k in self.metrics_at_k['map']:
            scores[f'map@{k}'] = self.compute_mean(self.average_precision, all_ground_truths, all_results, k)
        for k in self.metrics_at_k['mrr']:
            scores[f'mrr@{k}'] = self.compute_mean(self.reciprocal_rank, all_ground_truths, all_results, k)
        for k in self.metrics_at_k['ndcg']:
            scores[f'ndcg@{k}'] = self.compute_mean(self.ndcg, all_ground_truths, all_results, k)
        return scores

    def compute_mean(self, metric_fn, ground_truths, results, k=None):
        return mean([metric_fn(gt, res, k) for gt, res in zip(ground_truths, results)])

    def recall(self, ground_truths, results, k=None):
        k = len(results) if k is None else k
        rel = [1 if d in ground_truths else 0 for d in results[:k]]
        return sum(rel) / len(ground_truths) if ground_truths else 0.0

    def reciprocal_rank(self, ground_truths, results, k=None):
        k = len(results) if k is None else k
        for i, d in enumerate(results[:k]):
            if d in ground_truths:
                return 1 / (i + 1)
        return 0.0

    def average_precision(self, ground_truths, results, k=None):
        k = len(results) if k is None else k
        ap = 0.0
        hits = 0
        for i, d in enumerate(results[:k]):
            if d in ground_truths:
                hits += 1
                ap += hits / (i + 1)
        return ap / len(ground_truths) if ground_truths else 0.0

    def ndcg(self, ground_truths, results, k=None):
        k = len(results) if k is None else k
        dcg = 0.0
        for i, d in enumerate(results[:k]):
            rel_i = 1 if d in ground_truths else 0
            if rel_i > 0:
                dcg += rel_i / log2(i + 2)  # i+2 because ranks start at 1
        # Compute ideal DCG
        ideal_rels = [1] * min(len(ground_truths), k)
        idcg = sum([rel / log2(idx + 2) for idx, rel in enumerate(ideal_rels)])
        return dcg / idcg if idcg > 0 else 0.0


def evaluate_language(lang):
    ranks_file = f"ranks/jina_reranked_{lang}.json"
    gold_file = f"gold_standard_{lang}.json"
    output_file = f"results/jina_rerank_eval_{lang}.txt"

    with open(ranks_file, encoding="utf-8") as f:
        ranked_results = json.load(f)

    with open(gold_file, encoding="utf-8") as f:
        gold_data = json.load(f)

    query_ids = sorted(ranked_results.keys(), key=int)

    all_results = []
    all_ground_truths = []

    for qid in tqdm(query_ids, desc=f"Preparing data for {lang.upper()}"):
        retrieved = ranked_results[qid]
        relevant = gold_data[qid]

        all_results.append(retrieved)
        all_ground_truths.append(relevant)

    evaluator = Evaluator(metrics_at_k=metrics_at_k)
    scores = evaluator.compute_all_metrics(all_results, all_ground_truths)

    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write(f"Jina Reranking Evaluation Results for language: {lang.upper()}\n")
        print(f"\nJina Reranking Evaluation Results for language: {lang.upper()}")
        for metric, value in scores.items():
            line = f"{metric}: {value:.4f}"
            print(line)
            f_out.write(line + "\n")

    print(f"\nResults for {lang.upper()} saved to {output_file}")


for lang in ["nl", "fr"]:
    evaluate_language(lang)