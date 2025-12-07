from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_DIR = "./merged-space-model"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
except OSError:
    print(f"Error: Model directory not found at '{MODEL_DIR}'.")
    print("Please ensure the model is correctly merged and located in the specified directory.")
    exit()


def ask_space(query):
    inputs = tokenizer(query, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    print("Chat with Oolit! Type 'exit', 'No', 'N', or 'bye' to quit.")
    while True:
        prompt = input("You: ")
        if prompt.lower() in ["exit", "no", "n", "bye"]:
            print("Goodbye!")
            break
        answer = ask_space(prompt)
        print(f"Oolit : {answer}")
