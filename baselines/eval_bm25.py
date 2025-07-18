import json
from statistics import mean
from tqdm import tqdm
import os

metrics_at_k = {'recall': [10, 50, 100, 200, 500], 'map': [100], 'mrr': [100]}


class Evaluator:
    def __init__(self, metrics_at_k):
        self.metrics_at_k = metrics_at_k

    def compute_all_metrics(self, all_results, all_ground_truths):
        scores = dict()
        for k in self.metrics_at_k['recall']:
            recall_scalar = self.compute_mean(self.recall, all_ground_truths, all_results, k)
            scores[f'recall@{k}'] = recall_scalar

        for k in self.metrics_at_k['map']:
            map_scalar = self.compute_mean(self.average_precision, all_ground_truths, all_results, k)
            scores[f'map@{k}'] = map_scalar

        for k in self.metrics_at_k['mrr']:
            mrr_scalar = self.compute_mean(self.reciprocal_rank, all_ground_truths, all_results, k)
            scores[f'mrr@{k}'] = mrr_scalar

        return scores

    def compute_mean(self, metric_fn, ground_truths, results, k=None):
        return mean([metric_fn(gt, res, k) for gt, res in zip(ground_truths, results)])

    def precision(self, ground_truths, results, k=None):
        k = len(results) if k is None else k
        rel = [1 if d in ground_truths else 0 for d in results[:k]]
        return sum(rel) / len(results[:k])

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


evaluator = Evaluator(metrics_at_k=metrics_at_k)
os.makedirs("results", exist_ok=True)

for lang in ["fr", "nl"]:
    bm25_file = f"ranks/bm25_ranked_results_{lang}.json"   # bm25_bsard_results_{language}.json , "ranks/bm25_ranked_results_{lang}.json"
    gold_file = f"gold/gold_standard_{lang}.json"
    output_file = f"results/bm25_eval_{lang}.txt"

    with open(bm25_file, encoding="utf-8") as f:
        bm25_results = json.load(f)

    with open(gold_file, encoding="utf-8") as f:
        gold_data = json.load(f)

    query_ids = sorted(bm25_results.keys(), key=int)

    all_results = []
    all_ground_truths = []

    for qid in tqdm(query_ids, desc=f"Preparing data for {lang.upper()}"):
        retrieved = bm25_results[qid]
        relevant = gold_data[qid]

        all_results.append(retrieved)
        all_ground_truths.append(relevant)

    scores = evaluator.compute_all_metrics(all_results, all_ground_truths)

    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write(f"\nBM25 Evaluation Results for language: {lang.upper()}\n")
        print(f"\nBM25 Evaluation Results for language: {lang.upper()}")
        for metric, value in scores.items():
            if metric in ["mrr@10", "map@10"]:
                continue
            line = f"{metric}: {value:.4f}"
            print(line)
            f_out.write(line + "\n")