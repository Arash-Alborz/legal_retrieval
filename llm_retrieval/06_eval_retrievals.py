import json
from sklearn.metrics import precision_recall_fscore_support


with open("../sampling_hard_negatives/gold_standard_nl.json") as f:
    gold_data = json.load(f)

with open("retrievals/json/qwen3.235B_id_retrieval_nl.json") as f:
    output_data = json.load(f)

per_query_scores = {}
total_tp = total_fp = total_fn = 0

for query_id in gold_data:
    gold_docs = set(gold_data.get(query_id, []))
    pred_docs = set(output_data.get(query_id, []))

    tp = len(gold_docs & pred_docs)
    fp = len(pred_docs - gold_docs)
    fn = len(gold_docs - pred_docs)

    total_tp += tp
    total_fp += fp
    total_fn += fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    per_query_scores[query_id] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }

# macro
macro_precision = sum(s["precision"] for s in per_query_scores.values()) / len(per_query_scores)
macro_recall = sum(s["recall"] for s in per_query_scores.values()) / len(per_query_scores)
macro_f1 = sum(s["f1"] for s in per_query_scores.values()) / len(per_query_scores)

# micro
micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

print("Macro-averaged metrics:")
print(f"Precision: {macro_precision:.4f}")
print(f"Recall:    {macro_recall:.4f}")
print(f"F1 Score:  {macro_f1:.4f}")

print("\nMicro-averaged metrics:")
print(f"Precision: {micro_precision:.4f}")
print(f"Recall:    {micro_recall:.4f}")
print(f"F1 Score:  {micro_f1:.4f}")

import os

os.makedirs("retrievals/evaluation", exist_ok=True)

with open("retrievals/evaluation/evaluation_id_retr_qwen3.235b.txt", "w") as f:
    f.write("Macro-averaged metrics:\n")
    f.write(f"Precision: {macro_precision:.4f}\n")
    f.write(f"Recall:    {macro_recall:.4f}\n")
    f.write(f"F1 Score:  {macro_f1:.4f}\n\n")

    f.write("Micro-averaged metrics:\n")
    f.write(f"Precision: {micro_precision:.4f}\n")
    f.write(f"Recall:    {micro_recall:.4f}\n")
    f.write(f"F1 Score:  {micro_f1:.4f}\n")