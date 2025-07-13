import requests
import json

REGISTRY_URL = "https://raw.githubusercontent.com/nishaero/ai-infra-samples/main/dynamic-llm-serving/models/model-info.json"

def fetch_model_info():
    resp = requests.get(REGISTRY_URL)
    resp.raise_for_status()
    return json.loads(resp.text)

def get_current_model():
    info = fetch_model_info()
    return info.get("model_id"), info

def switch_model(new_model_id, user_token=None):
    # In production, this would update the model-info.json in GitHub via API (requires PAT)
    # For demo, print action and instruct manual update.
    print(f"Switching model to: {new_model_id}")
    return True
