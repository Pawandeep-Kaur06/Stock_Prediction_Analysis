import os
import requests
from dotenv import load_dotenv

# Load your API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Could not find GEMINI_API_KEY")
else:
    print("Asking Google what models you have access to...")
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        print("\n--- AVAILABLE MODELS ---")
        for model in data.get('models', []):
            # We only care about models that can generate text
            if 'generateContent' in model.get('supportedGenerationMethods', []):
                # Print the exact string you need to use
                print(model['name'].replace('models/', ''))
        print("------------------------\n")
    except Exception as e:
        print(f"Error checking models: {e}")