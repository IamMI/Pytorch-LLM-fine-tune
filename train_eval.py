import torch
from tqdm import tqdm

def train(dataloader, model, optimizer, device, save_path, num_epochs: int=3):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_train_loss}")
        
        if save_path:
            model.save_pretrained(save_path)
            
def eval(dataloader, model, device):
    model.eval()
    eval_loss = 0
    correct = 0
    total = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

        eval_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        labels = batch["labels"]
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_eval_loss = eval_loss / len(dataloader)
    accuracy = correct / total
    # print('-'*9 + 'Eval Finish' + '-'*9)
    # print(f"Evaluation Loss: {avg_eval_loss}")
    # print(f"Accuracy: {accuracy}")
    return avg_eval_loss, accuracy
    
    