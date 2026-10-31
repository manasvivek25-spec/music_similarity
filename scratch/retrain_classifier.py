import json
from src.origin_classifier import run_origin_classifier_suite

print("=== Training Origin Classifier on Expanded 261-Track Corpus ===")
res = run_origin_classifier_suite()
print("Status:", res.get("status"))
print("Best Model:", res.get("best_model"), "Best F1:", res.get("best_f1"))
print("Total Samples:", res.get("total_samples"), "Human:", res.get("human_samples"), "AI:", res.get("ai_samples"))

group_eval = res.get("group_aware_evaluation", {})
print("\nGroup-Aware Evaluation:")
for grp, data in group_eval.get("group_performance", {}).items():
    print(f"  Group '{grp}': Human Recall={data.get('human_recall')}, AI Recall={data.get('ai_recall')}, Mean P(AI)={data.get('mean_prob_ai')}")

logo = group_eval.get("leave_one_generator_out", {})
print("\nLeave-One-Generator-Out (LOGO) Generalization:")
for gen, data in logo.items():
    print(f"  Held-out '{gen}': Unseen AI Recall={data.get('unseen_generator_recall')}, Human Recall={data.get('held_out_human_recall')}, Test Size={data.get('test_sample_size')}")

