# DetectGPT Module & Function Map

This diagram shows the relationships between the main modules and functions in the DetectGPT codebase, with a focus on the core logic for text perturbation, likelihood computation, and evaluation.

```mermaid
graph TD
    subgraph Data Preparation
        A1[generate_data]
        A2[generate_samples]
        A3[strip_newlines / trim_to_shorter_length / truncate_to_substring]
    end
    subgraph Perturbation
        B1[perturb_texts]
        B2[perturb_texts_]
        B3[tokenize_and_mask]
        B4[replace_masks]
        B5[extract_fills]
        B6[apply_extracted_fills]
    end
    subgraph Scoring & Metrics
        C1[get_lls / get_ll]
        C2[get_likelihood]
        C3[get_rank]
        C4[get_entropy]
        C5[get_roc_metrics / get_precision_recall_metrics]
    end
    subgraph Evaluation
        D1[get_perturbation_results]
        D2[run_perturbation_experiment]
        D3[run_baseline_threshold_experiment]
        D4[eval_supervised]
    end
    subgraph Visualization
        E1[save_roc_curves]
        E2[save_ll_histograms]
        E3[save_llr_histograms]
    end

    %% Data flow
    A1 --> A2
    A2 --> D1
    D1 --> B1
    B1 --> B2
    B2 --> B3
    B3 --> B4
    B4 --> B5
    B5 --> B6
    B6 --> D1
    D1 --> C1
    C1 --> D2
    D2 --> E1
    D2 --> E2
    D2 --> E3
    D1 --> D3
    D1 --> D4
    C1 --> C5
    C5 --> D2

    %% Supporting functions
    A1 -.-> A3
    A2 -.-> A3
```

**Legend:**
- **Data Preparation:** Loading, cleaning, and sampling data
- **Perturbation:** Masking and filling text to create perturbed versions
- **Scoring & Metrics:** Computing likelihoods, ranks, entropy, and evaluation metrics
- **Evaluation:** Running experiments to distinguish real vs. AI-generated text
- **Visualization:** Plotting and saving results
- Dashed arrows indicate supporting/helper functions

---
This map helps you see how the main components interact and where the core logic for text alteration and evaluation resides.
