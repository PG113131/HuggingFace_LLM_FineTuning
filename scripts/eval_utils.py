import torch

def evaluate(model, dataloader, accelerator):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += len(batch["labels"])
    return correct / total
