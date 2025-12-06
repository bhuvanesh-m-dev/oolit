from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_DIR = "./merged-space-model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

def ask_space(query):
    inputs = tokenizer(query, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# Example
print(ask_space("What is the Sun made of?"))
