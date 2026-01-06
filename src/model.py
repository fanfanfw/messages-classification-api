import torch
import torch.nn as nn
from transformers import AutoModel


class MultiTaskClassifier(nn.Module):
    def __init__(self, model_name='indolem/indobert-base-uncased', num_labels=2, num_priorities=3, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.label_classifier = nn.Linear(hidden_size, num_labels)
        self.priority_classifier = nn.Linear(hidden_size, num_priorities)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        
        label_logits = self.label_classifier(pooled_output)
        priority_logits = self.priority_classifier(pooled_output)
        
        return label_logits, priority_logits


class MultiTaskLoss(nn.Module):
    def __init__(self, label_weight=1.0, priority_weight=1.0, priority_class_weights=None):
        super().__init__()
        self.label_weight = label_weight
        self.priority_weight = priority_weight
        self.label_loss_fn = nn.CrossEntropyLoss()
        self.priority_loss_fn = nn.CrossEntropyLoss(
            weight=priority_class_weights if priority_class_weights is not None else None
        )
    
    def forward(self, label_logits, priority_logits, label_targets, priority_targets):
        label_loss = self.label_loss_fn(label_logits, label_targets)
        priority_loss = self.priority_loss_fn(priority_logits, priority_targets)
        total_loss = self.label_weight * label_loss + self.priority_weight * priority_loss
        return total_loss, label_loss, priority_loss
