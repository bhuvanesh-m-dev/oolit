# test_space_lora.py
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
import torch

LORA_PATH = "./content/outputs/space-lora"
BASE_MODEL = "distilgpt2"   # change if needed
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer from:", LORA_PATH)
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, use_fast=True)

# Ensure tokenizer has pad token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token or "<|pad|>"})
    print("Added pad token:", tokenizer.pad_token)

print("Loading base model:", BASE_MODEL)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
# Resize tokens to align tokenizer (you already did this earlier; harmless to do again)
base.resize_token_embeddings(len(tokenizer))

print("Loading LoRA adapter from:", LORA_PATH)
model = PeftModel.from_pretrained(base, LORA_PATH)
model.to(device)
model.eval()
print("Model & adapter loaded. Device:", device)

# Build a safer generation config
gen_config = {
    "max_new_tokens": 200,
    "do_sample": True,
    "temperature": 0.8,
    "top_p": 0.92,
    "top_k": 50,
    "repetition_penalty": 1.05,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
    "bos_token_id": tokenizer.bos_token_id,
    # "stopping_criteria": ... (advanced, optional)
}

def ask_raw(prompt, gen_cfg=gen_config):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Use generation_config API for newer transformers
    generation_config = GenerationConfig(**gen_cfg)
    out = model.generate(**inputs, generation_config=generation_config)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text

def ask(prompt, prefix="Q: ", answer_marker="\nA:"):
    # Use a small Q/A template which many fine-tuned Q/A models expect
    full_prompt = f"{prefix}{prompt}{answer_marker} "
    raw = ask_raw(full_prompt)
    # Remove the prompt part and return only the newly generated answer (if present)
    if raw.startswith(full_prompt):
        return raw[len(full_prompt):].strip()
    # fallback: if model echoed prompt, try splitting on the answer_marker
    if answer_marker in raw:
        return raw.split(answer_marker, 1)[1].strip()
    return raw.strip()

if __name__ == "__main__":
    questions = [
        "What is a nebula?",
        "What is the Sun made of?",
        "Can people live on Mars?"
    ]

    for q in questions:
        print("\n---")
        print("Q:", q)
        ans = ask(q)
        if not ans:
            print("[Model returned empty answer — try increasing max_new_tokens or merging the adapter]")
        else:
            print("A:", ans)
