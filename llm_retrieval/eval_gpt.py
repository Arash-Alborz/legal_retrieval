import json
from pathlib import Path
from tqdm import tqdm
from statistics import mean

predictions_json = Path("retrievals/json/gpt4.1.mini_pw_retrievals_nl.json")
gold_json = Path("gold_data/gold_standard_nl.json")
output_dir = Path("evaluation")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "eval_pw_gpt4.1_mini_nl.txt"

ks = [1, 5, 10, 20, 50, 100]

# load predictions
with open(predictions_json, encoding="utf-8") as f:
    predictions = json.load(f)

# load gold
with open(gold_json, encoding="utf-8") as f:
    gold = json.load(f)

print(f"Loaded predictions for {len(predictions)} queries")
print(f"Loaded gold standard for {len(gold)} queries")

metrics = {f"R@{k}": [] for k in ks}
metrics.update({f"MRR@{k}": [] for k in ks})
metrics.update({f"MAP@{k}": [] for k in ks})

def compute_hits(relevant, predicted, k):
    hits = [1 if doc in relevant else 0 for doc in predicted[:k]]
    return hits

for qid in tqdm(predictions.keys(), desc="Evaluating"):
    pred = predictions[qid]
    gold_set = set(gold[qid])

    for k in ks:
        hits = compute_hits(gold_set, pred, k)

        recall = sum(hits) / len(gold_set) if gold_set else 0.0
        metrics[f"R@{k}"].append(recall)

        rr = 0.0
        for rank, h in enumerate(hits):
            if h:
                rr = 1.0 / (rank + 1)
                break
        metrics[f"MRR@{k}"].append(rr)

        ap = 0.0
        hit_count = 0
        for rank, h in enumerate(hits):
            if h:
                hit_count += 1
                ap += hit_count / (rank + 1)
        ap /= len(gold_set) if gold_set else 1
        metrics[f"MAP@{k}"].append(ap)

print()
with open(output_file, "w", encoding="utf-8") as out:
    for m, vals in metrics.items():
        line = f"{m}: {mean(vals):.4f}"
        print(line)
        out.write(line + "\n")

print(f"\nEvaluation results saved to: {output_file}")