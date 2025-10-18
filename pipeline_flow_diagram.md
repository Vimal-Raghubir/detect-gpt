# DetectGPT Pipeline Flow Diagram

This diagram outlines the order of function calls in `run.py`, starting from the main method. It shows the high-level flow and when each major function is invoked during a typical experiment run.

```mermaid
graph TD
    A[Start: Main Method] --> B[Parse Arguments & Setup]
    B --> C[Load Data: generate_data]
    C --> D[Generate Samples: generate_samples]
    D --> E[Load Base Model & Tokenizer: load_base_model_and_tokenizer]
    E --> F[Sample from Model: sample_from_model]
    F --> G[Run Perturbation Experiment?]
    G -- Yes --> H[get_perturbation_results]
    H --> I[perturb_texts]
    I --> J[perturb_texts_]
    J --> K[tokenize_and_mask]
    K --> L[replace_masks]
    L --> M[extract_fills]
    M --> N[apply_extracted_fills]
    N --> O[Compute Log Likelihoods: get_lls]
    O --> P[run_perturbation_experiment]
    P --> Q[save_roc_curves / save_ll_histograms]
    G -- No --> R[Run Baseline Experiment: run_baseline_threshold_experiment]
    R --> S[get_lls / get_rank / get_entropy]
    S --> Q
    Q --> T[End]

    %% Supporting functions
    subgraph Supporting
        U[strip_newlines]
        V[trim_to_shorter_length]
        W[truncate_to_substring]
    end
    C -.-> U
    D -.-> V
    D -.-> W
```

**How to read this diagram:**
- The flow starts at the main method and proceeds through argument parsing, data loading, and model setup.
- Depending on the experiment type, it either runs a perturbation experiment (left branch) or a baseline experiment (right branch).
- Each box represents a function call or major step, with arrows showing the order of execution.
- Dashed arrows indicate supporting functions that are called as helpers within the main flow.

---

This diagram provides a high-level overview. For more detail, refer to the function explanations in `explanations.md`.
