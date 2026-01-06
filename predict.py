import argparse
import torch
from transformers import AutoTokenizer
from pathlib import Path

from src.model import MultiTaskClassifier


class MessageClassifier:
    def __init__(self, model_path='model_output'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = Path(model_path)
        
        checkpoint = torch.load(model_path / 'best_model.pt', map_location=self.device, weights_only=False)
        self.model_name = checkpoint['model_name']
        self.label_map = checkpoint['label_map']
        self.priority_map = checkpoint['priority_map']
        
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        self.inv_priority_map = {v: k for k, v in self.priority_map.items()}
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path / 'tokenizer')
        self.model = MultiTaskClassifier(model_name=self.model_name)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text, return_probs=False):
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            label_logits, priority_logits = self.model(input_ids, attention_mask)
        
        label_probs = torch.softmax(label_logits, dim=1)
        priority_probs = torch.softmax(priority_logits, dim=1)
        
        label_idx = label_logits.argmax(dim=1).item()
        priority_idx = priority_logits.argmax(dim=1).item()
        
        result = {
            'label': self.inv_label_map[label_idx],
            'priority': self.inv_priority_map[priority_idx],
            'label_confidence': label_probs[0][label_idx].item(),
            'priority_confidence': priority_probs[0][priority_idx].item()
        }
        
        if return_probs:
            result['label_probs'] = {self.inv_label_map[i]: p.item() for i, p in enumerate(label_probs[0])}
            result['priority_probs'] = {self.inv_priority_map[i]: p.item() for i, p in enumerate(priority_probs[0])}
        
        return result
    
    def predict_batch(self, texts):
        return [self.predict(text) for text in texts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model_output')
    parser.add_argument('--text', type=str, help='Single text to classify')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    args = parser.parse_args()
    
    classifier = MessageClassifier(args.model_path)
    
    if args.text:
        result = classifier.predict(args.text, return_probs=True)
        print(f"\nText: {args.text}")
        print(f"Label: {result['label']} (confidence: {result['label_confidence']:.2%})")
        print(f"Priority: {result['priority']} (confidence: {result['priority_confidence']:.2%})")
    
    elif args.interactive:
        print("Interactive mode. Type 'quit' to exit.\n")
        while True:
            text = input("Enter text: ").strip()
            if text.lower() == 'quit':
                break
            if not text:
                continue
            result = classifier.predict(text)
            print(f"  -> Label: {result['label']}, Priority: {result['priority']}")
            print(f"     Confidence: label={result['label_confidence']:.2%}, priority={result['priority_confidence']:.2%}\n")
    
    else:
        test_texts = [
            "pesanan saya belum sampai, sudah 3 hari",
            "ada promo hari ini?",
            "refund saya kapan masuk?",
            "produk warna hitam ready stock?"
        ]
        print("Sample predictions:\n")
        for text in test_texts:
            result = classifier.predict(text)
            print(f"Text: {text}")
            print(f"  -> Label: {result['label']}, Priority: {result['priority']}\n")


if __name__ == '__main__':
    main()
