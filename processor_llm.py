import requests
import re

OLLAMA_MODEL = "llama3"  # or whatever model name you've pulled locally

def classify_with_ollama(log_msg):
    prompt = f"""Classify the log message into one of these categories: 
    (1) Workflow Error, (2) Deprecation Warning.
    If you can't figure out a category, use "Unclassified".
    Put the category inside <category> </category> tags. 
    Log message: {log_msg}"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False  # We want the full completion at once
        }
    )

    if response.status_code != 200:
        raise Exception(f"Ollama returned an error: {response.text}")

    result = response.json()
    content = result.get("response", "")

    match = re.search(r'<category>(.*?)</category>', content, flags=re.DOTALL)
    category = "Unclassified"
    if match:
        category = match.group(1).strip()

    return category

if __name__ == "__main__":
    print(classify_with_ollama("Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active."))
    print(classify_with_ollama("The 'ReportGenerator' module will be retired in version 4.0. Please migrate to the 'AdvancedAnalyticsSuite' by Dec 2025"))
    print(classify_with_ollama("System reboot initiated by user 12345."))
    print(classify_with_ollama("Hi babe"))
