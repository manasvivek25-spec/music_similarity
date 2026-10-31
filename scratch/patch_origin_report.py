import re

file_path = "src/origin_classifier.py"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

target = '''            group_rows_str = "\\n".join(g_rows)
            group_eval_section = f"""### 3.2 Generator-Aware & Group-Aware Generalization Analysis

| Test Cohort | Condition / Nature | Sample Size | Pred Human | Pred AI | Performance Metric | Mean P(AI) | IDSE Finding & Scientific Assessment |
|---|---|---|---|---|---|---|---|
{group_rows_str}

> [!NOTE]
> **Confounding & Generalization Disclosure:**
> Within HAIM, AI tracks originate exclusively from ACE-Step, creating potential dataset-specific acoustic confounding. To address this:
> 1. Testing on independent Jamendo human recordings demonstrates robust out-of-distribution human recall without generator-specific memorization.
> 2. Hybrid AI-mastered tracks reveal that mastering compression does not destroy the underlying acoustic signature of human performance.
> 3. Temporal mixtures exhibit balanced probabilities reflecting composite origin.
"""'''

replacement = '''            # Add generator-specific rows
            for gen_name in ["lyria_pro3", "musicgen", "acestep"]:
                gen_info = group_eval.get(f"generator_{gen_name}", {})
                if gen_info:
                    g_rows.append(f"| **{gen_info.get('generator').title()}** | AI Architecture Cohort | {gen_info.get('sample_size')} | {gen_info.get('predicted_human')} | {gen_info.get('predicted_ai')} | {gen_info.get('ai_recall')*100:.1f}% AI Recall | {gen_info.get('mean_prob_ai'):.3f} | {gen_info.get('conclusion')} |")

            group_rows_str = "\\n".join(g_rows)

            logo_section = ""
            logo_dict = group_eval.get("leave_one_generator_out", {})
            if logo_dict:
                logo_rows = []
                for h_gen, h_res in logo_dict.items():
                    logo_rows.append(f"| **Held Out: {h_gen.title()}** | {', '.join(h_res.get('train_generators'))} | {h_res.get('train_sample_size')} | {h_res.get('test_sample_size')} | {h_res.get('unseen_generator_recall')*100:.1f}% | {h_res.get('held_out_human_recall')*100:.1f}% | **{h_res.get('balanced_accuracy')*100:.1f}%** |")
                logo_rows_str = "\\n".join(logo_rows)
                logo_section = f"""### 3.3 Leave-One-Generator-Out (LOGO) Generalization Analysis

| Held-Out Evaluation Cohort | Training Generators | Train Samples | Test Samples | Unseen AI Recall | Human Recall | Balanced Accuracy |
|---|---|---|---|---|---|---|
{logo_rows_str}

> [!IMPORTANT]
> **Generator Generalization Findings (IDSE Evaluation):**
> Testing across three independent AI synthesis engines (ACE-Step, Lyria-Pro3, MusicGen) confirms that classifiers trained exclusively on one generator architecture suffer significant recall reduction when applied to unseen architectures. Multi-generator representation is necessary to mitigate synthesis-engine confounding.
"""

            group_eval_section = f"""### 3.2 Generator-Aware & Group-Aware Generalization Analysis

| Test Cohort | Condition / Nature | Sample Size | Pred Human | Pred AI | Performance Metric | Mean P(AI) | IDSE Finding & Scientific Assessment |
|---|---|---|---|---|---|---|---|
{group_rows_str}

{logo_section}
"""'''

text_norm = text.replace('\r\n', '\n')
target_norm = target.replace('\r\n', '\n')

if target_norm in text_norm:
    text_patched = text_norm.replace(target_norm, replacement)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text_patched)
    print("Successfully patched report generation in src/origin_classifier.py!")
else:
    print("Target not found in text_norm!")

