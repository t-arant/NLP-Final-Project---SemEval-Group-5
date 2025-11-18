import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel
from tqdm import tqdm

import torch

from bioes_classification import balanced_accuracy_ignore_pads, LABEL_TO_NUM, NUM_LABELS

class CombinedDataset(Dataset):
    '''
    This might be super wrong. 
    Tried combining datasets - will work on this more tomorrow.
    '''
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.sentences = dataframe["Text"].tolist()
        self.aspects = dataframe["Aspect"].tolist()
        self.bioes_labels = dataframe["BIOES"].tolist()
        self.va_labels = dataframe[["Valence", "Arousal"]].values.astype(float)  # VA scores
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        text = f"{self.aspects[idx]}: {self.sentences[idx]}"
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length", 
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        bioes_numeric = [LABEL_TO_NUM[tag] for tag in self.bioes_labels[idx]]
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "bioes_labels": torch.tensor(bioes_numeric, dtype=torch.long),  # For BIOES
            "va_labels": torch.tensor(self.va_labels[idx], dtype=torch.float)  # For regression
        }

class CombinedModel(nn.Module):
    '''
    A BERT-based regressor for predicting Valence and Arousal scores,
    combined with a BIOES-tagging classifier. 

    - Uses a pretrained BERT backbone to encode text.
    - Takes the [CLS] token representation as sentence-level embedding.
    - Adds a dropout layer and a linear head to output 2 values: [Valence, Arousal].
    - DOES NOT INCLUDE helper methods for one training epoch and one evaluation epoch. 

    Args:
        model_name (str): HuggingFace model name, default "bert-base-multilingual-cased".
        dropout (float): Dropout rate before the regression head.
    '''
    def __init__(self, model_name="distilbert-base-multilingual-cased", dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # 2 heads
        self.bioes_head = nn.Linear(self.backbone.config.hidden_size, NUM_LABELS)  # BIOES
        self.reg_head = nn.Linear(self.backbone.config.hidden_size, 2)   # valence/arousal

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # BIOES -> all tkns 
        bioes_logits = self.bioes_head(hidden_states)
        
        cls_output = hidden_states[:, 0]  # first token = [CLS]
        cls_output = self.dropout(cls_output)
        va_scores = self.reg_head(cls_output)
        
        return bioes_logits, va_scores
    
# instead of model.train_epoch()
def train_combined_epoch(model, dataloader, optimizer, bioes_loss_fn, va_loss_fn, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        bioes_labels = batch["bioes_labels"].to(device) 
        va_labels = batch["va_labels"].to(device) 

        optimizer.zero_grad()
        
        bioes_logits, va_scores = model(input_ids, attention_mask)
        
        # two loss fns 
        bioes_loss = bioes_loss_fn(bioes_logits.view(-1, NUM_LABELS), bioes_labels.view(-1))
        va_loss = va_loss_fn(va_scores, va_labels)
        
        # combine losses
        total_batch_loss = bioes_loss + va_loss
        
        total_batch_loss.backward()
        optimizer.step()
        total_loss += total_batch_loss.item()
    
    return total_loss / len(dataloader)

# instead of model.eval_epoch()
def eval_combined_epoch(model, dataloader, bioes_loss_fn, va_loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bioes_labels = batch["bioes_labels"].to(device)
            va_labels = batch["va_labels"].to(device)

            bioes_logits, va_scores = model(input_ids, attention_mask)
            
            bioes_loss = bioes_loss_fn(bioes_logits.view(-1, NUM_LABELS), bioes_labels.view(-1))
            va_loss = va_loss_fn(va_scores, va_labels)
            
            total_batch_loss = bioes_loss + va_loss
            total_loss += total_batch_loss.item()
    
    return total_loss / len(dataloader)

# instead of model_inference_bal_acc
def model_inference_combined(model, dataloader, device):
    model.eval()
    bioes_accuracies = []
    va_errors = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        bioes_labels = batch["bioes_labels"]
        va_labels = batch["va_labels"]

        with torch.no_grad():
            bioes_logits, va_scores = model(input_ids, attention_mask)

        # (copied from model_inference_bal_acc)
        bioes_preds = bioes_logits.argmax(dim=-1)
        for index in range(bioes_preds.shape[0]):
            acc = balanced_accuracy_ignore_pads(bioes_labels[index], bioes_preds[index], -100)
            bioes_accuracies.append(acc)
        
        # VA regression error (new)
        va_error = torch.nn.functional.mse_loss(va_scores, va_labels.to(device))
        va_errors.append(va_error.item())

    avg_bioes_acc = sum(bioes_accuracies) / len(bioes_accuracies)
    avg_va_error = sum(va_errors) / len(va_errors)
    
    return avg_bioes_acc, avg_va_error
