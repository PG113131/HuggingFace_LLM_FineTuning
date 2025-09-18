from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model_and_tokenizer(checkpoint="distilbert-base-uncased", num_labels=2):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
    return model, tokenizer

def save_model_and_tokenizer(model, tokenizer, output_dir="imdb-distilbert-finetuned"):
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
