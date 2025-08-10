import json
from pathlib import Path
import math
from tqdm import tqdm
from statistics import mean

#predictions_json = Path("retrievals/json/gemini_2.5_flash_id_retrieval_nl.json")              # id_retr:                      gemini 2.5. Flash
#predictions_json = Path("retrievals/json/gemini_2.5_flash_pro_id_retrieval_nl.json")          # id_retr:                      gemini 2.5. pro

#predictions_json = Path("retrievals/json/gpt4.1.mini_id_retrieval_nl.json")                   # id_retr:                      gpt 4.1 - mini
#predictions_json = Path("retrievals/json/gpt4o.mini_id_retrieval_nl.json")                    # id_retr:                      gpt 4o - mini

#predictions_json = Path("retrievals/json/llama3.3.70b_id_retrieval_nl.json")                  # id_retr:                      llama3.3.70b
predictions_json = Path("retrievals/json/llama4.scout_id_retrieval_nl.json")                  # id_retr:                      llama4.scout(17Bx16B)
#predictions_json = Path("retrievals/json/qwen3-235B_id_retrieval_nl.json")                    # id_retr:                      qwen3-235B

#predictions_json = Path("retrievals/json/gpt4.1.mini_bin_class_retrieval_nl.json")            # binary_classification_retr:   gpt 4.1 - mini
#predictions_json = Path("retrievals/json/gpt4o.mini_bin_class_retrieval_nl.json")             # binary_classification_retr:   gpt 4.1 - mini

#predictions_json = Path("retrievals/json/llama3.3.70b_bin_class_retrieval_nl.json")           # binary_classification_retr:   llama3.3-70b
predictions_json = Path("retrievals/json/llama4.scout_bin_class_retrieval_nl.json")           # binary_classification_retr:   llama4.scout(17Bx16B)
#predictions_json = Path("retrievals/json/qwen3.235b_bin_class_retrieval_nl.json")              # binary_classification_retr:   qwen3-236b

#predictions_json = Path("retrievals/json/gemini_2.5.flash_bin_class_retrieval_nl.json") 



gold_json = Path("gold_data/gold_standard_nl.json")
output_dir = Path("evaluation")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "eval_binary_retrieval_ppllama4.scout.txt"                 # eval txt name template: [  eval_id_retrieval_model.txt  ] --> change model name
                                                                                  # eval txt name template: [  eval_binary_retrieval_model.txt  ] --> change model name

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
metrics.update({f"nDCG@{k}": [] for k in ks})

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

        dcg = 0.0
        for i, h in enumerate(hits):
            if h:
                dcg += 1.0 / (math.log2(i + 2))  # +2 because log2(rank+1), and rank = 0-based

        ideal_hits = [1] * min(len(gold_set), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(len(ideal_hits)))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        metrics[f"nDCG@{k}"].append(ndcg)

print()
with open(output_file, "w", encoding="utf-8") as out:
    for m, vals in metrics.items():
        line = f"{m}: {mean(vals):.4f}"
        print(line)
        out.write(line + "\n")

print(f"\nEvaluation results saved to: {output_file}")