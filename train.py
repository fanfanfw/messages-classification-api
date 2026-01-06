import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, confusion_matrix
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
    all_label_preds = []
    all_label_targets = []
    all_priority_preds = []
    all_priority_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            priorities = batch['priority'].to(device)
            
            label_logits, priority_logits = model(input_ids, attention_mask)
            loss, _, _ = loss_fn(label_logits, priority_logits, labels, priorities)
            
            total_loss += loss.item()
            label_preds = label_logits.argmax(dim=1)
            priority_preds = priority_logits.argmax(dim=1)
            correct_labels += (label_preds == labels).sum().item()
            correct_priorities += (priority_preds == priorities).sum().item()
            total += labels.size(0)

            all_label_preds.extend(label_preds.detach().cpu().tolist())
            all_label_targets.extend(labels.detach().cpu().tolist())
            all_priority_preds.extend(priority_preds.detach().cpu().tolist())
            all_priority_targets.extend(priorities.detach().cpu().tolist())
    
    label_f1_macro = f1_score(all_label_targets, all_label_preds, labels=[0, 1], average='macro', zero_division=0)
    priority_f1_macro = f1_score(all_priority_targets, all_priority_preds, labels=[0, 1, 2], average='macro', zero_division=0)
    priority_cm = confusion_matrix(all_priority_targets, all_priority_preds, labels=[0, 1, 2])

    return (
        total_loss / len(dataloader),
        correct_labels / total,
        correct_priorities / total,
        float(label_f1_macro),
        float(priority_f1_macro),
        priority_cm,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        type=str,
        nargs='+',
        default=['dataset_combined.csv', 'dataset_malay.csv'],
        help='One or more CSV files with columns: text,label,priority',
    )
    parser.add_argument('--model_name', type=str, default='bert-base-multilingual-cased')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default='model_output')
    parser.add_argument('--checkpoint_name', type=str, default='best_model.pt')
    parser.add_argument('--label_weight', type=float, default=1.0)
    parser.add_argument('--priority_weight', type=float, default=2.0)
    parser.add_argument(
        '--selection_metric',
        type=str,
        choices=['combined_acc', 'priority_f1_macro', 'combined_f1_macro'],
        default='priority_f1_macro',
        help='Metric to decide which checkpoint is best',
    )
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=2,
        help='Stop if selection metric does not improve for N epochs (0 to disable)',
    )
    parser.add_argument(
        '--print_confusion_on_best',
        action='store_true',
        help='Print priority confusion matrix when a new best checkpoint is saved',
    )
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    texts, labels, priorities = load_data(args.data)
    print(f"Loaded {len(texts)} samples")
    
    stratify_key = [f"{l}__{p}" for l, p in zip(labels, priorities)]
    train_texts, val_texts, train_labels, val_labels, train_priorities, val_priorities = train_test_split(
        texts, labels, priorities, test_size=0.2, random_state=42, stratify=stratify_key
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
    loss_fn = MultiTaskLoss(
        label_weight=args.label_weight,
        priority_weight=args.priority_weight,
        priority_class_weights=priority_weights,
    )
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)
    
    best_metric = -1e9
    epochs_since_improve = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for epoch in range(args.epochs):
        train_loss, train_label_acc, train_priority_acc = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        (
            val_loss,
            val_label_acc,
            val_priority_acc,
            val_label_f1_macro,
            val_priority_f1_macro,
            val_priority_cm,
        ) = evaluate(model, val_loader, loss_fn, device)
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Label Acc: {train_label_acc:.4f}, Priority Acc: {train_priority_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Label Acc: {val_label_acc:.4f}, Priority Acc: {val_priority_acc:.4f}")
        print(f"  Val F1 (macro): Label: {val_label_f1_macro:.4f}, Priority: {val_priority_f1_macro:.4f}")
        
        if args.selection_metric == 'combined_acc':
            metric = (val_label_acc + val_priority_acc) / 2
        elif args.selection_metric == 'combined_f1_macro':
            metric = (val_label_f1_macro + val_priority_f1_macro) / 2
        else:
            metric = val_priority_f1_macro

        if metric > best_metric:
            best_metric = metric
            epochs_since_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_name': args.model_name,
                'label_map': {'issues': 0, 'search': 1},
                'priority_map': {1: 0, 2: 1, 3: 2}
            }, output_dir / args.checkpoint_name)
            tokenizer.save_pretrained(output_dir / 'tokenizer')
            print(f"  Saved best model ({args.selection_metric}: {metric:.4f})")
            if args.print_confusion_on_best:
                print("  Priority confusion matrix (rows=true, cols=pred; classes=0/1/2 => prio 1/2/3):")
                print(val_priority_cm)
        else:
            epochs_since_improve += 1

        if args.early_stopping_patience > 0 and epochs_since_improve >= args.early_stopping_patience:
            print(f"Early stopping: no improvement in {args.selection_metric} for {args.early_stopping_patience} epochs")
            break
    
    print(f"\nTraining complete. Best model saved to {output_dir}")


if __name__ == '__main__':
    main()
