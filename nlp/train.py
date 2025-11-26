import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    BertForTokenClassification,
    BertTokenizerFast,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from datasets import load_dataset
from seqeval.metrics import f1_score, classification_report
import argparse
import random
import numpy as np
from tqdm import tqdm
import pandas as pd
import ast

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_marker(marker_str):
    """Parse marker string to extract character positions"""
    if marker_str == '[]' or not marker_str:
        return []
    try:
        return ast.literal_eval(marker_str)
    except:
        return []

def create_ner_tags(text, markers):
    """Convert text and character markers to NER BIO tags"""
    tokens = text.split()
    ner_tags = ['O'] * len(tokens)
    
    if not markers:
        return tokens, ner_tags
    
    # Sort markers by start position
    markers = sorted(markers, key=lambda x: x[0])
    
    # Create list of tokens with their character positions
    token_positions = []
    char_pos = 0
    for token in tokens:
        start = char_pos
        end = char_pos + len(token)
        token_positions.append((start, end))
        char_pos = end + 1  # +1 for space
    
    # For each marker, find corresponding tokens
    for start, end in markers:
        entity_tokens = []
        
        # Find all tokens that overlap with the marker
        for i, (token_start, token_end) in enumerate(token_positions):
            if not (token_end <= start or token_start >= end):
                entity_tokens.append(i)
        
        # Assign BIO tags to found tokens
        if entity_tokens:
            for j, token_idx in enumerate(entity_tokens):
                if j == 0:  # First token gets B- tag
                    ner_tags[token_idx] = 'B-MOUNTAIN'
                else:       # Subsequent tokens get I- tags
                    ner_tags[token_idx] = 'I-MOUNTAIN'
    
    return tokens, ner_tags

def preprocess_dataset(csv_path):
    """Load CSV dataset and convert to NER format"""
    df = pd.read_csv(csv_path)
    processed_data = []
    
    for _, row in df.iterrows():
        text = row['text']
        markers = parse_marker(row['marker'])
        
        # Convert character markers to token-level NER tags
        tokens, ner_tags = create_ner_tags(text, markers)
        
        # Skip empty sequences
        if tokens:
            processed_data.append({
                'tokens': tokens,
                'ner_tags': ner_tags
            })
    
    return processed_data

class MountainNERDataset(torch.utils.data.Dataset):
    """Custom dataset for Mountain NER task"""
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = {'O': 0, 'B-MOUNTAIN': 1, 'I-MOUNTAIN': 2}
        self.id2label = {0: 'O', 1: 'B-MOUNTAIN', 2: 'I-MOUNTAIN'}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item['tokens']
        labels = [self.label2id[tag] for tag in item['ner_tags']]

        # Tokenize text and align labels with subword tokens
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Align labels with tokenizer output (handle subword tokens)
        word_ids = encoding.word_ids()
        aligned_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens ([CLS], [SEP], [PAD]) get -100 (ignored in loss)
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                # First token of a word gets the original label
                aligned_labels.append(labels[word_idx])
            else:
                # Subsequent subword tokens get -100
                aligned_labels.append(-100)
            previous_word_idx = word_idx

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Backward pass
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, id2label):
    """Evaluate model on validation set"""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Model inference
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get predictions
            logits = outputs.logits
            preds = torch.argmax(logits, dim=2)
            
            # Convert to label strings and filter out special tokens
            for pred_seq, label_seq, mask_seq in zip(preds, labels, attention_mask):
                pred_labels = []
                true_label = []
                
                for p, l, m in zip(pred_seq, label_seq, mask_seq):
                    if m.item() == 1 and l.item() != -100:  # Only non-padding and non-special tokens
                        pred_labels.append(id2label[p.item()])
                        true_label.append(id2label[l.item()])
                
                if true_label:
                    predictions.append(pred_labels)
                    true_labels.append(true_label)
    
    # Detailed statistics for analysis
    mountain_true_tokens = sum(1 for seq in true_labels for tag in seq if tag != 'O')
    mountain_pred_tokens = sum(1 for seq in predictions for tag in seq if tag != 'O')
    sequences_with_mountains = sum(1 for seq in true_labels if any(tag != 'O' for tag in seq))
    
    print(f"\nDETAILED STATISTICS")
    print(f"Total sequences: {len(true_labels)}")
    print(f"Sequences with mountains: {sequences_with_mountains}")
    print(f"Real mountain tokens: {mountain_true_tokens}")
    print(f"Predicted mountain tokens: {mountain_pred_tokens}")
    
    # Show examples of model predictions
    print(f"\nEXAMPLES WITH MOUNTAINS")
    mountain_examples = 0
    for i, (true, pred) in enumerate(zip(true_labels, predictions)):
        if any(tag != 'O' for tag in true):
            print(f"Example {mountain_examples + 1}:")
            print(f"  True: {true}")
            print(f"  Pred: {pred}")
            mountain_examples += 1
            if mountain_examples >= 3:
                break
    
    try:
        # Calculate NER metrics using seqeval
        f1 = f1_score(true_labels, predictions)
        report = classification_report(true_labels, predictions)
        
        # Additional metrics for detailed analysis
        from seqeval.metrics import precision_score, recall_score
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        
        print(f"\nENTITY LEVEL METRICS")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")
        
    except Exception as e:
        print(f"Error in metrics calculation: {e}")
        f1 = 0.0
        report = "Error generating report"
    
    return f1, report

def split_dataset(data, train_ratio=0.8):
    """Split dataset into train and validation sets"""
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


def main(args):
    """Main training function"""
    set_seed(42)

    # Load and preprocess dataset
    print("Loading and preprocessing dataset...")
    all_data = preprocess_dataset(args.dataset_path)

    # Split into train/validation
    train_data, val_data = split_dataset(all_data, train_ratio=0.8)

    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    
    # Define label mappings for NER task
    label_list = ['O', 'B-MOUNTAIN', 'I-MOUNTAIN']
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    
    model = BertForTokenClassification.from_pretrained(
        'bert-base-cased',
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )
    
    # Create datasets and data loaders
    train_dataset = MountainNERDataset(train_data, tokenizer, max_len=args.max_len)
    val_dataset = MountainNERDataset(val_data, tokenizer, max_len=args.max_len)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        num_workers=2
    )
    
    # Setup optimizer and device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    best_f1 = 0.0
    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*50}")
        
        # Train and evaluate
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_f1, val_report = evaluate(model, val_loader, device, id2label)
        
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Validation F1: {val_f1:.4f}")
        print("\nClassification Report:")
        print(val_report)
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            print(f"\nNew best F1: {best_f1:.4f} - Saving model...")
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
    
    print(f"\n{'='*50}")
    print(f"Training completed! Best F1: {best_f1:.4f}")
    print(f"{'='*50}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Mountain NER model')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to mountain dataset CSV')
    parser.add_argument('--output_dir', type=str, default='./models/mountain_ner_model', help='Output directory for model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--max_len', type=int, default=128, help='Max sequence length')
    
    args = parser.parse_args()
    main(args)