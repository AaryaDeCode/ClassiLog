import joblib
from sentence_transformers import SentenceTransformer

# Global variables for lazy loading
model_embedding = None
model_classification = None

def get_models():
    global model_embedding, model_classification
    if model_embedding is None:
        model_embedding = SentenceTransformer('all-MiniLM-L6-v2')  # Lazy load embedding model
    if model_classification is None:
        model_classification = joblib.load("models/log_classifier.joblib")
    return model_embedding, model_classification

def classify_with_bert(log_message):
    model_embedding, model_classification = get_models()
    embeddings = model_embedding.encode([log_message])
    probabilities = model_classification.predict_proba(embeddings)[0]
    
    if max(probabilities) < 0.5:
        return "Unclassified"
    
    predicted_label = model_classification.predict(embeddings)[0]
    return predicted_label

if __name__ == "__main__":
    logs = [
        "alpha.osapi_compute.wsgi.server - 12.10.11.1 - API returned 404 not found error",
        "GET /v2/3454/servers/detail HTTP/1.1 RCODE   404 len: 1583 time: 0.1878400",
        "System crashed due to drivers errors when restarting the server",
        "Hey bro, chill ya!",
        "Multiple login failures occurred on user 6454 account",
        "Server A790 was restarted unexpectedly during the process of data transfer"
    ]
    for log in logs:
        label = classify_with_bert(log)
        print(log, "->", label)
