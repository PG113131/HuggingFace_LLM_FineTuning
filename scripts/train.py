from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from accelerate import Accelerator
import torch

from scripts.dataset_utils import load_and_tokenize_dataset
from scripts.model_utils import load_model_and_tokenizer, save_model_and_tokenizer
from scripts.eval_utils import evaluate

# 1. Config
checkpoint = "distilbert-base-uncased"
batch_size = 16
epochs = 3
lr = 5e-5
max_length = 256

# 2. Accelerator
accelerator = Accelerator()

# 3. Load model + tokenizer
model, tokenizer = load_model_and_tokenizer(checkpoint)

# 4. Dataset
tokenized_datasets = load_and_tokenize_dataset(tokenizer, max_length=max_length)
train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=batch_size)

# 5. Optimizer + Scheduler
optimizer = AdamW(model.parameters(), lr=lr)
num_training_steps = len(train_dataloader) * epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# 6. Prepare for accelerate
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

# 7. Training Loop
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
    accuracy = evaluate(model, eval_dataloader, accelerator)
    accelerator.print(f"Epoch {epoch+1} | Accuracy: {accuracy:.4f}")

# 8. Save Model
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
save_model_and_tokenizer(unwrapped_model, tokenizer, "models/imdb-distilbert-finetuned")
