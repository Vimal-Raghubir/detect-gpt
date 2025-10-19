# DetectGPT Setup & Execution Guide

This guide will help you set up and run the DetectGPT script, including all dependencies, required models, and key configuration steps.

---

## 1. Prerequisites
- **Python 3.7+** (recommended: Python 3.8 or 3.9)
- **pip** (Python package manager)
- **Git** (optional, for cloning repositories)
- **CUDA** (if using GPU acceleration)

## 2. Install Dependencies
1. Open a terminal in the `detect-gpt` directory.
2. Install all required Python packages:

```powershell
pip install -r requirements.txt
```

## 3. Required Models
DetectGPT uses HuggingFace Transformers models and optionally the OpenAI API. By default, you need:

- **Base Model:** (for scoring and sampling)
    - Default: `gpt2-medium` (HuggingFace)
    - Other options: Any GPT-2 variant, or a custom model supported by Transformers
- **Mask-Filling Model:** (for perturbation)
    - Default: `t5-large` (HuggingFace)
    - Other options: Any T5 variant (e.g., `t5-base`, `t5-3b`)

### How to Download Models
The script will automatically download models the first time you run it, if you have an internet connection. To pre-download:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration
# Download base model
AutoModelForCausalLM.from_pretrained('gpt2-medium')
AutoTokenizer.from_pretrained('gpt2-medium')
# Download mask-filling model
T5ForConditionalGeneration.from_pretrained('t5-large')
AutoTokenizer.from_pretrained('t5-large')
```
Or, simply run the script and let it download as needed.

### Using OpenAI API
If you want to use OpenAI models (e.g., `text-davinci-003`):
- Set `--openai_model text-davinci-003` and provide your API key with `--openai_key YOUR_KEY`.
- No local model download is needed for OpenAI API usage.

## 4. Running the Script
From the `detect-gpt` directory, run:

```powershell
python run.py
```

To customize, add arguments (see `run.py` for all options):

For initial run:
```powershell
python detectGPT.py --dataset "EdinburghNLP/xsum" --dataset_key document --base_model_name gpt2-medium --mask_filling_model_name t5-large --n_samples 2000 --batch_size 10 --perturb_cache_out "Dataset/Perturbed"
```

USE THIS FOR EVERY RUN:
```powershell
python detectGPT.py --dataset "EdinburghNLP/xsum" --dataset_key document --base_model_name gpt2-medium --mask_filling_model_name t5-large --n_samples 2000 --batch_size 10 --data_json "Dataset/AI/AI_EdinburghNLP_xsum.json" --perturb_cache_out "Dataset/Perturbed"
```

## 5. Key Configuration Options
- `--dataset` and `--dataset_key`: Choose your dataset and the key to use (e.g., `xsum`, `document`)
- `--base_model_name`: Name of the base model (default: `gpt2-medium`)
- `--mask_filling_model_name`: Name of the mask-filling model (default: `t5-large`)
- `--openai_model` and `--openai_key`: Use OpenAI API instead of local models
- `--n_samples`: Number of samples to process
- `--span_length`, `--pct_words_masked`: Perturbation parameters

## 6. Output
- Results, plots, and logs are saved in a timestamped folder in the working directory.

## 7. Troubleshooting
- **Model Download Issues:** Ensure you have a stable internet connection for first-time downloads.
- **CUDA/Device Errors:** Check your PyTorch and CUDA installation if using GPU.
- **Missing Packages:** Re-run `pip install -r requirements.txt` if you see import errors.

---

For more details, see the function explanations and diagrams in this project.
