from transformers import pipeline

# Load the trained model and tokenizer
model_path = "imdb-distilbert-finetuned"
sentiment_analyzer = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

# Example inference
new_review = "This movie was absolutely fantastic! I loved every moment of it."
result = sentiment_analyzer(new_review)
print(f"Review: {new_review}")
print(f"Predicted sentiment: {result[0]['label']} with score {result[0]['score']:.4f}")

new_review_2 = "This movie was terrible, I hated it."
result_2 = sentiment_analyzer(new_review_2)
print(f"Review: {new_review_2}")
print(f"Predicted sentiment: {result_2[0]['label']} with score {result_2[0]['score']:.4f}")