# This file contains the code for defining, training, and evaluating a BERT-based model for subtask 1 
# DM 11/28/2025 - automates some hyperparameter tuning (hidden dims, activation fn)

import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel
from tqdm import tqdm

import torch

from root_mse import rmse

CLS_SIZE = 768

class VADataset(Dataset):
    '''
    This class takes in DataFrame structured training data, and tokenizes the text for use with Encoder Transformer models
    
    Args:
    - dataframe (DataFrame): Dataframe of Text to be tokenized 
    - tokenizer (AutoTokenizer): Hugging Face Autotokenizer 
    - max_len (int): Max number of tokens per tokenization

    Class attributes:
    - sentences (list) - list of string sentences
    - aspect (list) - list of string aspect labels 
    - tokenizer (AutoTokenizer): Hugging Face Autotokenizer 
    - max_len (int): Max number of tokens per tokenization

    Returns: 
    Tokenized Data 
        - input ids (torch tensor): vectorized tokens of input sentences
        - attention mask (torch tensor): mask of true and padding tokens 
    VA_labels (torch tensor: float) - training labels of valence arousal scorce pairs 
    '''
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.sentences = dataframe["Text"].tolist()
        self.aspects = dataframe["Aspect"].tolist()
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
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "va_labels": torch.tensor(self.va_labels[idx], dtype=torch.float)  # For regression
        }

class RegressorModel(nn.Module):
    '''
    A BERT-based regressor for predicting Valence and Arousal scores. 

    - Uses a pretrained BERT backbone to contextualize text embeddings.
    - Takes the [CLS] token representation as sentence-level embedding.
    - Runs [CLS] token through dropout layer and regression head.
    - DOES NOT INCLUDE helper methods for one training epoch and one evaluation epoch.
    - Activation function testing can be automated, but stays the same for each layer. 

    Args:
        model_name (str): HuggingFace model name, default "distilbert-base-multilingual-cased".
        dropout (float): Dropout rate before the regression head.
        hidden_dims (list): List of hidden layer dimensions (e.g., [384] for one hidden layer)
        activation (str): Activation function - 'relu', 'leaky_relu', 'gelu', 'elu', or 'none'

    (This version automates hyperparameter tuning, call with 
    model = RegressorModel(hidden_dims=[YOUR_DIMS], activation='YOUR_ACTIVATION'))
    '''
    def __init__(self, model_name="distilbert-base-multilingual-cased", dropout=0.1, 
                 hidden_dims=[384], activation='none'):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        layers = []
        input_dim = CLS_SIZE
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 2))
        
        self.reg_head = nn.Sequential(*layers)
    
    def _get_activation(self, activation):
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.3),
            'gelu': nn.GELU(),
            'elu': nn.ELU(),
            'none': nn.Identity()  # No activation
        }
        return activations.get(activation, nn.Identity())
 
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state        
        cls_output = hidden_states[:, 0]  # first token = [CLS]
        cls_output = self.dropout(cls_output)
        va_scores = self.reg_head(cls_output)
        
        return va_scores
    
    def freeze_regressor(self):
        for parameter in self.reg_head.parameters():
            parameter.requires_grad = False
    
    def unfreeze_regressor(self):
        for parameter in self.reg_head.parameters():
            parameter.requires_grad = True
    
    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
    
    def unfreeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

# Training function for model 
# Input parameters: HuggingFace model, dataloader of training data, optimizer function, loss function, and device (GPU if available)
# Trains model and catelogs loss. 
# Returns averaged loss across batches   
def train_regressor_epoch(model, dataloader, optimizer, va_loss_fn, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        va_labels = batch["va_labels"].to(device) 

        optimizer.zero_grad()
        
        va_scores = model(input_ids, attention_mask)
        
        va_loss = va_loss_fn(va_scores, va_labels)
        
        va_loss.backward()
        optimizer.step()
        total_loss += va_loss.item()
    
    return total_loss / len(dataloader)

# Evaluation function for regressor - used in Training 
def eval_regressor_epoch(model, dataloader, va_loss_fn, device):
    model.eval()

    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            va_labels = batch["va_labels"].to(device)

            va_scores = model(input_ids, attention_mask)
            
            va_loss = va_loss_fn(va_scores, va_labels)
            
            total_batch_loss =  va_loss
            total_loss += total_batch_loss.item()
    
    return total_loss / len(dataloader)

# Function for evaluating model after training
# Returns average of error for whole dataset provided 
def model_inference(model, loss_fn, dataloader, device):
    model.eval()
    va_errors = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        va_labels = batch["va_labels"]

        with torch.no_grad():
            va_scores = model(input_ids, attention_mask)
        
        # VA regression error (new)
        va_error = loss_fn(va_scores, va_labels.to(device))
        va_errors.append(va_error.item())

    avg_va_error = sum(va_errors) / len(va_errors)
    
    return avg_va_error

# Function used to generate JSONL predictions based on input data
# Outputs structured python list in JSONL format
def generate_results(model, dataloader, device, dataframe):
    model.eval()

    predictions_list = []
    combined_preds = []

    model.freeze_regressor()
    model.freeze_backbone()

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        with torch.no_grad():
            va_scores = model(input_ids, attention_mask)
        
        batch_results_list = va_scores.cpu().tolist()
        predictions_list.extend(batch_results_list)

    for pred in predictions_list:
        score1 = f"{pred[0]:.2f}"
        score2 = f"{pred[1]:.2f}"
        combined_preds.append(f"{score1}#{score2}")

    dataframe['VA'] = combined_preds

    nested = [
        {
            "ID": group_name,
            "Aspect_VA": group[["Aspect", "VA"]].to_dict(orient="records")
        }
        for group_name, group in dataframe.groupby("ID")
    ]
    
    return nested
        