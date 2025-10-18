# DetectGPT Core Logic Summary

This document summarizes the core logic of DetectGPT, focusing on how the codebase perturbs text using AI models and evaluates whether a text is AI-generated or not.

---

## 1. Data Preparation
- **Functions:** `generate_data`, `generate_samples`, `strip_newlines`, `trim_to_shorter_length`, `truncate_to_substring`
- **Role:** Load, clean, and format datasets. Prepare both original and sampled (AI-generated) texts for analysis.

## 2. Text Perturbation (Alteration)
- **Functions:** `perturb_texts`, `perturb_texts_`, `tokenize_and_mask`, `replace_masks`, `extract_fills`, `apply_extracted_fills`
- **Role:**
    - Randomly mask out spans in the text.
    - Use a mask-filling model (e.g., T5) or random fills to generate plausible replacements for masked spans.
    - Reconstruct perturbed versions of the original and sampled texts.
- **Why:** Perturbed texts are used to probe the model's likelihood landscape and reveal differences between real and generated text.

## 3. Likelihood & Metric Computation
- **Functions:** `get_ll`, `get_lls`, `get_likelihood`, `get_rank`, `get_entropy`
- **Role:**
    - Compute log-likelihoods, token ranks, and entropy for original, sampled, and perturbed texts using the base model.
    - These metrics quantify how "surprising" or "typical" a text is under the model.

## 4. Evaluation & Discrimination
- **Functions:** `get_perturbation_results`, `run_perturbation_experiment`, `run_baseline_threshold_experiment`, `eval_supervised`, `get_roc_metrics`, `get_precision_recall_metrics`
- **Role:**
    - Compare metrics for real vs. generated texts.
    - Use ROC and precision-recall curves to evaluate how well the method distinguishes between the two.
    - Run both perturbation-based and baseline experiments for comparison.

## 5. Visualization & Reporting
- **Functions:** `save_roc_curves`, `save_ll_histograms`, `save_llr_histograms`
- **Role:**
    - Plot and save visualizations of results for analysis and presentation.

---

## Core Logic for Detecting AI-Generated Text
1. **Prepare Data:** Load and clean real texts, generate AI samples.
2. **Perturb Texts:** Mask and fill spans to create perturbed versions of both real and generated texts.
3. **Score Texts:** Compute likelihoods and other metrics for all versions.
4. **Evaluate:** Use statistical tests and metrics to see if real and generated texts can be separated based on their scores.
5. **Visualize:** Plot results to interpret and present findings.

---

**Key Insight:**
- DetectGPT leverages the difference in model likelihoods (and related metrics) between original and perturbed texts to distinguish real from AI-generated content. Real texts tend to have more stable likelihoods under perturbation, while generated texts show greater changes, enabling effective discrimination.

For detailed function explanations, see `explanations.md`. For the pipeline flow, see `pipeline_flow_diagram.md`. For module relationships, see `module_function_map.md`.
