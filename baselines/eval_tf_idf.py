import json
import os
from statistics import mean
from tqdm import tqdm

os.makedirs("results", exist_ok=True)

metrics_at_k = {
    'recall': [10, 50, 100, 200, 500],
    'map': [100],
    'mrr': [100]
}

class Evaluator:
    def __init__(self, metrics_at_k):
        self.metrics_at_k = metrics_at_k

    def compute_all_metrics(self, all_results, all_ground_truths):
        scores = dict()
        for k in self.metrics_at_k['recall']:
            scores[f'recall@{k}'] = self._mean(self.recall, all_ground_truths, all_results, k)
        for k in self.metrics_at_k['map']:
            scores[f'map@{k}'] = self._mean(self.average_precision, all_ground_truths, all_results, k)
        for k in self.metrics_at_k['mrr']:
            scores[f'mrr@{k}'] = self._mean(self.reciprocal_rank, all_ground_truths, all_results, k)
        return scores

    def _mean(self, metric_fn, gts, res, k=None):
        return mean([metric_fn(gt, r, k) for gt, r in zip(gts, res)])

    def recall(self, gt, res, k=None):
        k = k or len(res)
        hits = sum(1 for d in res[:k] if d in gt)
        return hits / len(gt) if gt else 0.0

    def reciprocal_rank(self, gt, res, k=None):
        k = k or len(res)
        for i, d in enumerate(res[:k]):
            if d in gt:
                return 1 / (i + 1)
        return 0.0

    def average_precision(self, gt, res, k=None):
        k = k or len(res)
        score, hits = 0.0, 0
        for i, d in enumerate(res[:k]):
            if d in gt:
                hits += 1
                score += hits / (i + 1)
        return score / len(gt) if gt else 0.0


# evaluating
evaluator = Evaluator(metrics_at_k)

for lang in ["fr", "nl"]:
    print(f"\nEvaluating language: {lang.upper()}")

    ranks_path = f"ranks/tfidf_ranked_results_{lang}.json"
    gold_path  = f"gold/gold_standard_{lang}.json"

    with open(ranks_path, encoding="utf-8") as f:
        ranks = json.load(f)
    with open(gold_path, encoding="utf-8") as f:
        gold = json.load(f)

    query_ids = sorted(ranks.keys(), key=int)

    all_results = []
    all_ground_truths = []

    for qid in tqdm(query_ids, desc=f"Preparing data {lang}"):
        retrieved = ranks[qid]
        relevant  = gold[qid]
        all_results.append(retrieved)
        all_ground_truths.append(relevant)

    scores = evaluator.compute_all_metrics(all_results, all_ground_truths)

    print(f"\nTF-IDF Evaluation Results for {lang.upper()}:")
    with open(f"results/tfidf_eval_{lang}.txt", "w", encoding="utf-8") as f_out:
        for metric, val in scores.items():
            line = f"{metric}: {val:.4f}"
            print(line)
            f_out.write(line + "\n")

    print(f"Results saved to: results/tfidf_eval_{lang}.txt")