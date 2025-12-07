from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

MODEL_DIR = "merged-space-model"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
except OSError:
    print(f"Error: Model directory not found at '{MODEL_DIR}'.")
    print("Please ensure the model is correctly merged and located in the specified directory.")
    exit()

# Ensure tokenizer has pad token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token or "<|pad|>"})
    print("Added pad token:", tokenizer.pad_token)

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
}

def ask_space(prompt, prefix="Q: ", answer_marker="\nA:"):
    # Use a small Q/A template which many fine-tuned Q/A models expect
    full_prompt = f"{prefix}{prompt}{answer_marker} "
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    
    generation_config = GenerationConfig(**gen_config)
    out = model.generate(**inputs, generation_config=generation_config)
    
    raw = tokenizer.decode(out[0], skip_special_tokens=True)
    
    # Remove the prompt part and return only the newly generated answer
    if raw.startswith(full_prompt):
        return raw[len(full_prompt):].strip()
    
    # Fallback if model didn't echo prompt exactly
    if answer_marker in raw:
        return raw.split(answer_marker, 1)[1].strip()
        
    return raw.strip()

if __name__ == "__main__":
    print("Chat with Oolit! Type 'exit', 'No', 'N', or 'bye' to quit.")
    while True:
        prompt = input("You: ")
        if prompt.lower() in ["exit", "no", "n", "bye"]:
            print("Goodbye!")
            break
        answer = ask_space(prompt)
        print(f"Oolit : {answer}")
