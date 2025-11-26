import torch
from transformers import BertForTokenClassification, BertTokenizerFast
import argparse

class MountainNER:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.model = BertForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.label_list = list(self.model.config.id2label.values())
        print(f"Loaded model with labels: {self.label_list}")
    
    def predict(self, text):
        """
        Predict Named Entities in text.
        Properly handles subword tokens (WordPiece tokenization).
        """
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=128,
            return_offsets_mapping=True,
            return_special_tokens_mask=True
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=2)

        # Get token information
        labels = predictions[0].cpu().numpy()
        word_ids = encoding.word_ids(batch_index=0)
        offsets = encoding['offset_mapping'][0].cpu().numpy()
        special_tokens_mask = encoding['special_tokens_mask'][0].cpu().numpy()

        # Group predictions by words
        # For each word, take the label of the FIRST subword token
        word_labels = {}
        word_offsets = {}
        
        for i, (word_id, label_id, is_special, offset) in enumerate(
            zip(word_ids, labels, special_tokens_mask, offsets)
        ):
            if is_special or word_id is None:
                continue
            
            label = self.model.config.id2label[label_id]
            
            # Take only the first subtoken of each word
            if word_id not in word_labels:
                word_labels[word_id] = label
                word_offsets[word_id] = {'start': offset[0], 'end': offset[1]}
            else:
                # Update end offset (for multi-subword words)
                word_offsets[word_id]['end'] = offset[1]

        entities = []
        current_entity_words = []
        
        for word_id in sorted(word_labels.keys()):
            label = word_labels[word_id]
            
            if label == 'B-MOUNTAIN':
                # Finish previous entity
                if current_entity_words:
                    entities.append(
                        self._create_entity(text, current_entity_words, word_offsets)
                    )
                
                # Start new entity
                current_entity_words = [word_id]
            
            elif label == 'I-MOUNTAIN':
                # Continue current entity (only if we have B-MOUNTAIN)
                if current_entity_words:
                    current_entity_words.append(word_id)
            
            elif label == 'O':
                # Finish current entity
                if current_entity_words:
                    entities.append(
                        self._create_entity(text, current_entity_words, word_offsets)
                    )
                    current_entity_words = []
        
        # Final entity
        if current_entity_words:
            entities.append(
                self._create_entity(text, current_entity_words, word_offsets)
            )

        return entities
    
    def _create_entity(self, text, word_ids, word_offsets):
        """
        Create entity dict from words.
        word_ids: list of word IDs that belong to the entity
        word_offsets: dict {word_id: {'start': ..., 'end': ...}}
        """
        # Take start of first word and end of last word
        start = int(word_offsets[word_ids[0]]['start'])
        end = int(word_offsets[word_ids[-1]]['end'])

        entity_text = text[start:end]
        
        return {
            "entity": entity_text,
            "type": "MOUNTAIN",
            "start": start,
            "end": end
        }

def main(args):
    ner = MountainNER(args.model_path)
    
    if args.text:
        print(f"\nText: {args.text}")
        entities = ner.predict(args.text)
        
        if entities:
            print("Found mountains:")
            for entity in entities:
                print(f"   - {entity['entity']}")
        else:
            print("No mountains found")
        
        print("─" * 60)
    
    elif args.input_file:
        print(f"Processing file: {args.input_file}\n")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    entities = ner.predict(line)
                    print(f"\n{i}. Text: {line}")
                    if entities:
                        print(f"   Mountains: {[e['entity'] for e in entities]}")
                    else:
                        print("   No mountains found")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mountain NER Inference')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained model')
    parser.add_argument('--text', type=str, 
                       help='Single text to analyze')
    parser.add_argument('--input_file', type=str, 
                       help='File with texts to analyze (one per line)')
    
    args = parser.parse_args()
    
    if not args.text and not args.input_file:
        parser.error("Provide --text or --input_file")
    
    main(args)