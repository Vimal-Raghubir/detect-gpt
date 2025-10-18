from transformers import AutoTokenizer, AutoModelForCausalLM
import os
print("HF_HOME", os.getenv("HF_HOME"))
print("TRANSFORMERS_CACHE", os.getenv("TRANSFORMERS_CACHE"))

# Will print actual cache paths being used and force-download if missing
tok = AutoTokenizer.from_pretrained("gpt2")
mdl = AutoModelForCausalLM.from_pretrained("gpt2")
print("Loaded gpt2 tokenizer and model OK")
