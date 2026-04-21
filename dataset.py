import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, BartTokenizer
from datasets import load_dataset
from config import MAX_LEN, BATCH_SIZE, EMOTION_TO_LABEL

def get_tokenizers():
    roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    return roberta_tokenizer, bart_tokenizer

class GoEmotionsDataset(Dataset):
    """ Used for Phase 1: Training the Emotion Classifier """
    def __init__(self, data, tokenizer, max_len=MAX_LEN):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        text = str(item['text'])
        label = item['label']

        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class EDGenerationDataset(Dataset):
    """ Used for Phase 2: Training the Generator (BART + Fusion) """
    def __init__(self, data, roberta_tokenizer, bart_tokenizer, max_len=MAX_LEN):
        self.data = data
        self.roberta_tokenizer = roberta_tokenizer
        self.bart_tokenizer = bart_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        full_context = str(item['context'])
        latest_user_utterance = str(item['latest_utterance'])
        bot_response = str(item['target'])

        # For RoBERTa Input (Needs only the immediate emotion)
        rob_inputs = self.roberta_tokenizer(
            latest_user_utterance,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # For BART Setup (Needs the whole history block)
        bart_inputs = self.bart_tokenizer(
            full_context,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        bart_targets = self.bart_tokenizer(
            bot_response,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        target_ids = bart_targets['input_ids'].squeeze(0)
        target_ids[target_ids == self.bart_tokenizer.pad_token_id] = -100

        return {
            'rob_input_ids': rob_inputs['input_ids'].squeeze(0),
            'rob_attention_mask': rob_inputs['attention_mask'].squeeze(0),
            'bart_input_ids': bart_inputs['input_ids'].squeeze(0),
            'bart_attention_mask': bart_inputs['attention_mask'].squeeze(0),
            'labels': target_ids
        }

def get_phase1_dataloaders(tokenizer, batch_size=BATCH_SIZE):
    print("Loading GoEmotions dataset for Phase 1...")
    dataset = load_dataset("go_emotions", "simplified", trust_remote_code=True)
    
    def process_split(split_name):
        processed = []
        for row in dataset[split_name]:
            # Using the first label if multiple exist for simplicity
            labels = row['labels']
            if len(labels) > 0:
                processed.append({'text': row['text'], 'label': labels[0]})
        return processed

    train_data = process_split('train')
    val_data = process_split('validation')
    
    train_dataset = GoEmotionsDataset(train_data, tokenizer)
    val_dataset = GoEmotionsDataset(val_data, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader

def get_phase2_dataloaders(roberta_tokenizer, bart_tokenizer, batch_size=BATCH_SIZE):
    print("Loading EmpatheticDialogues dataset for Phase 2...")
    raw_dataset = load_dataset("empathetic_dialogues", trust_remote_code=True)
    
    def prepare_dialogues(split):
        data = raw_dataset[split]
        processed_data = []
        conversations = {}
        for row in data:
            conv_id = row['conv_id']
            if conv_id not in conversations:
                conversations[conv_id] = []
            conversations[conv_id].append(row)
            
        for conv_id, turns in conversations.items():
            context_history = []
            for i in range(len(turns) - 1):
                user_text = turns[i]['utterance'].replace("_comma_", ",")
                bot_text = turns[i+1]['utterance'].replace("_comma_", ",")
                
                # Append user prompt to history
                context_history.append(f"User: {user_text}")
                
                # Create history string explicitly capped to last 6 elements
                # Use </s> separator so BART recognizes boundaries and stops copying history
                full_context = " </s> ".join(context_history[-6:])
                
                if user_text and bot_text:
                    processed_data.append({
                        'context': full_context,
                        'target': bot_text,
                        'latest_utterance': user_text
                    })
                
                # Add bot response so the next user loop knows what it is responding to
                context_history.append(f"Bot: {bot_text}")
                
        return processed_data

    train_data = prepare_dialogues('train')
    val_data = prepare_dialogues('validation')
    
    train_dataset = EDGenerationDataset(train_data, roberta_tokenizer, bart_tokenizer)
    val_dataset = EDGenerationDataset(val_data, roberta_tokenizer, bart_tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader
