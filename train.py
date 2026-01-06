import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from pathlib import Path

from src.model import MultiTaskClassifier, MultiTaskLoss
from src.dataset import MultiTaskDataset, load_data


def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, device):
    model.train()
    total_loss = 0
    correct_labels = 0
    correct_priorities = 0
    total = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        priorities = batch['priority'].to(device)
        
        optimizer.zero_grad()
        label_logits, priority_logits = model(input_ids, attention_mask)
        loss, _, _ = loss_fn(label_logits, priority_logits, labels, priorities)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        correct_labels += (label_logits.argmax(dim=1) == labels).sum().item()
        correct_priorities += (priority_logits.argmax(dim=1) == priorities).sum().item()
        total += labels.size(0)
    
    return total_loss / len(dataloader), correct_labels / total, correct_priorities / total


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct_labels = 0
    correct_priorities = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            priorities = batch['priority'].to(device)
            
            label_logits, priority_logits = model(input_ids, attention_mask)
            loss, _, _ = loss_fn(label_logits, priority_logits, labels, priorities)
            
            total_loss += loss.item()
            correct_labels += (label_logits.argmax(dim=1) == labels).sum().item()
            correct_priorities += (priority_logits.argmax(dim=1) == priorities).sum().item()
            total += labels.size(0)
    
    return total_loss / len(dataloader), correct_labels / total, correct_priorities / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='dataset_combined.csv')
    parser.add_argument('--model_name', type=str, default='indolem/indobert-base-uncased')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default='model_output')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    texts, labels, priorities = load_data(args.data)
    print(f"Loaded {len(texts)} samples")
    
    train_texts, val_texts, train_labels, val_labels, train_priorities, val_priorities = train_test_split(
        texts, labels, priorities, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    train_dataset = MultiTaskDataset(train_texts, train_labels, train_priorities, tokenizer, args.max_length)
    val_dataset = MultiTaskDataset(val_texts, val_labels, val_priorities, tokenizer, args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    priority_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=np.array([train_dataset.priority_map[p] for p in train_priorities]))
    priority_weights = torch.tensor(priority_weights, dtype=torch.float32).to(device)
    print(f"Priority class weights: {priority_weights}")
    
    model = MultiTaskClassifier(model_name=args.model_name).to(device)
    loss_fn = MultiTaskLoss(priority_class_weights=priority_weights)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)
    
    best_val_acc = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for epoch in range(args.epochs):
        train_loss, train_label_acc, train_priority_acc = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        val_loss, val_label_acc, val_priority_acc = evaluate(model, val_loader, loss_fn, device)
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Label Acc: {train_label_acc:.4f}, Priority Acc: {train_priority_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Label Acc: {val_label_acc:.4f}, Priority Acc: {val_priority_acc:.4f}")
        
        combined_acc = (val_label_acc + val_priority_acc) / 2
        if combined_acc > best_val_acc:
            best_val_acc = combined_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_name': args.model_name,
                'label_map': {'issues': 0, 'search': 1},
                'priority_map': {1: 0, 2: 1, 3: 2}
            }, output_dir / 'best_model.pt')
            tokenizer.save_pretrained(output_dir / 'tokenizer')
            print(f"  Saved best model (combined acc: {combined_acc:.4f})")
    
    print(f"\nTraining complete. Best model saved to {output_dir}")


if __name__ == '__main__':
    main()
