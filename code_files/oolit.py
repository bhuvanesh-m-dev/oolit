# oolit.py

import sys
from api import api

def oolit():
    # Create Ollama API instance
    api = OllamaAPI(base_url="http://localhost:11434")
    
    # Get prompt from command-line arguments
    if len(sys.argv) > 1:
        prompt = ' '.join(sys.argv[1:])
    else:
        # Provide a default prompt if none is given
        prompt_user = input("Enter your prompt : ")
        prompt = prompt_user
    
    # Send prompt to Ollama and get response
    print("Thinking...")
    response = api.send_prompt(prompt)
    
    # Print the response
    print(f"Oolit: {response}")

oolit()
