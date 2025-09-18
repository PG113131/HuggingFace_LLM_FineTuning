from datasets import load_dataset

def load_and_tokenize_dataset(tokenizer, max_length=256):
    dataset = load_dataset("imdb")

    def tokenize_function(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")
    return tokenized_datasets
