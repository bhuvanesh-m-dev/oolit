# merge_lora.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

LORA_DIR = "/home/bhuvanesh-m-ubuntu/Desktop/ML/space-lora/content/outputs/space-lora"
BASE_MODEL = "distilgpt2"
OUT_DIR = "/home/bhuvanesh-m-ubuntu/Desktop/ML/space-lora/merged-space-model"

print("Loading tokenizer from:", LORA_DIR)
tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
tok_len = len(tokenizer)
print("Tokenizer length:", tok_len)

print("Loading base model:", BASE_MODEL)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

print("Resizing base model token embeddings to tokenizer length...")
base.resize_token_embeddings(tok_len)

print("Loading LoRA adapter...")
peft_model = PeftModel.from_pretrained(base, LORA_DIR, torch_dtype=base.dtype)

print("Merging LoRA into base model (this may take time)...")
merged = peft_model.merge_and_unload()  # merges weights and unloads PEFT wrappers

print("Saving merged model to:", OUT_DIR)
merged.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print("Done. Merged model saved.")
