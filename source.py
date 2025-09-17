from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
import torch

# 1. Load dataset
dataset = load_dataset("imdb")

# 2. Tokenizer
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(
        example["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=256
    )

# 3. Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 4. Fix columns
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")

# 5. DataLoaders
train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=16, shuffle=True)
eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=16)

# 6. Model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# 7. Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# 8. Accelerator
accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

# 9. Scheduler
num_training_steps = len(train_dataloader) * 3   # 3 epochs
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# 10. Training Loop
epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    # --- Evaluation ---
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in eval_dataloader:
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += len(batch["labels"])

    accuracy = correct / total
    accelerator.print(f"Epoch {epoch+1} | Accuracy: {accuracy:.4f}")

# 11. Save model
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained("imdb-distilbert-finetuned")
tokenizer.save_pretrained("imdb-distilbert-finetuned")
