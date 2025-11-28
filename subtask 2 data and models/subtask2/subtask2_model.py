# This file contains the code for defining, training, and evaluating a BERT-based model for subtask 2 
# DM 11/28/2025 - same mods as in subtask 1 model 

import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel
from transformers import AutoTokenizer
from tqdm import tqdm

import torch

import numpy as np
from torchcrf import CRF
from root_mse import rmse

CLS_SIZE = 768

NUM_LABELS = 10


LABEL_TO_NUM = {
    "EMB_P": 0,        # use 0 as padding

    "EMB_O": 1,

    "ASP_B": 2,
    "ASP_I": 3,
    "ASP_E": 4,
    "ASP_S": 5,

    "OPN_B": 6,
    "OPN_I": 7,
    "OPN_E": 8,
    "OPN_S": 9,
}

class VA_ASP_OPN_Dataset(Dataset):
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
        self.opinions = dataframe["Opinion"].tolist()
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
            "va_labels": torch.tensor(self.va_labels[idx], dtype=torch.float),  # For regression
            "bioes_labels": torch.tensor(bioes_numeric, dtype=torch.long) # For BIOES
        }

class BERT_Tag_Reg_Model(nn.Module):
    '''
    A BERT-based multi-task model for predicting Valence/Arousal scores and BIOES tagging.

    - Uses a pretrained BERT backbone to contextualize text embeddings.
    - Takes the [CLS] token representation as sentence-level embedding.
    - Runs [CLS] token through dropout layer and regression head
    - DOES NOT INCLUDE helper methods for one training epoch and one evaluation epoch. 
    - Uses full sequence for BIOES tagging with CRF.
    - Configurable regression head architecture.

    Args:
        model_name (str): HuggingFace model name, default "distilbert-base-multilingual-cased".
        dropout (float): Dropout rate before the regression head.
        hidden_dims (list): List of hidden layer dimensions for regression head
        activation (str): Activation function - 'relu', 'leaky_relu', 'gelu', 'elu', or 'none'
    '''
    def __init__(self, model_name="distilbert-base-multilingual-cased", dropout=0.1, 
                 hidden_dims=[384], activation='none'):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.linear_classifier = nn.Linear(self.backbone.config.hidden_size, NUM_LABELS) # transform 
        self.crf_layer = CRF(NUM_LABELS, batch_first=True)
        
        # Build regression head dynamically
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

        x = self.dropout(hidden_states)
        linear_output = self.linear_classifier(x)
        
        return va_scores, linear_output
    
    def freeze_regressor(self):
        for parameter in self.reg_head.parameters():
            parameter.requires_grad = False
    
    def unfreeze_regressor(self):
        for parameter in self.reg_head.parameters():
            parameter.requires_grad = True

    def freeze_crf(self):
        for parameter in self.crf_layer.parameters():
            parameter.requires_grad = False
    
    def unfreeze_crf(self):
        for parameter in self.crf_layer.parameters():
            parameter.requires_grad = True

    def freeze_linear_classifier(self):
        for parameter in self.linear_classifier.parameters():
            parameter.requires_grad = False

    def unfreeze_linear_classifier(self):
        for parameter in self.linear_classifier.parameters():
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
        
        va_scores, lin_output = model(input_ids, attention_mask)
        
        va_loss = va_loss_fn(va_scores, va_labels)
        
        va_loss.backward()
        optimizer.step()
        total_loss += va_loss.item()
    
    return total_loss / len(dataloader)

def train_tagger_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tag_labels = batch["bioes_labels"].to(device)
        optimizer.zero_grad()

        va_scores, lin_output = model(input_ids, attention_mask)
        # outputs = outputs.view(-1, outputs.shape[-1])  # [B*seq_len, num_labels]
        # labels = labels.view(-1)  
        # labels, mask = adjust_mask(input_ids, labels, tokenizer)

        loss = -model.crf_layer(lin_output, tag_labels, mask=attention_mask.bool())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
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

            va_scores, lin_output = model(input_ids, attention_mask)
            
            va_loss = va_loss_fn(va_scores, va_labels)
            
            total_batch_loss =  va_loss
            total_loss += total_batch_loss.item()
    
    return total_loss / len(dataloader)

def eval_tagger_epoch(model, dataloader, device):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                tag_labels = batch["bioes_labels"].to(device)

                va_scores, lin_outputs = model(input_ids, attention_mask)

                # outputs = outputs.view(-1, outputs.shape[-1])  # [B*seq_len, num_labels]
                # labels = labels.view(-1)  

                loss = -model.crf_layer(lin_outputs, tag_labels, mask=attention_mask.bool())
                total_loss += loss.item()
        return total_loss / len(dataloader)

# Function for evaluating model after training
# Returns average of error for whole dataset provided 
def model_inference(model, loss_fn, dataloader, device):
    model.eval()
    va_errors = []
    tag_errors = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        va_labels = batch["va_labels"].to(device)
        tag_labels = batch["bioes_labels"].to(device)
        

        with torch.no_grad():
            va_scores, lin_output  = model(input_ids, attention_mask)
        
        # VA regression error (new)
        va_error = loss_fn(va_scores, va_labels.to(device))
        va_errors.append(va_error.item())

        loss = -model.crf_layer(lin_output, tag_labels, mask=attention_mask.bool())
        tag_errors.append(loss.item())

    avg_va_error = sum(va_errors) / len(va_errors)
    avg_tag_error = sum(tag_errors) / len(tag_errors)
    
    return avg_va_error, avg_tag_error

def balanced_accuracy_ignore_pads(y_true, y_pred, ignore_label=None):
    y_true_np = y_true.numpy()
    y_pred_np = np.array(y_pred)

    
    if ignore_label is not None:
        mask = y_true_np != ignore_label
        y_true_np = y_true_np[mask]
        # y_pred_np = y_pred_np[mask]
    
    len_label = len(y_true_np)
    y_pred_np = y_pred_np[0:len_label]
   

    if len(y_true_np) == 0:
        return 0  # edge case: all padding
    return balanced_accuracy_no_sklearn(y_true_np, y_pred_np)

def balanced_accuracy_no_sklearn(y_true, y_pred):
    """
    Compute balanced accuracy for 1D arrays of labels (no sklearn).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(y_true)
    recalls = []

    for c in classes:
        # True positives and false negatives for class c
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        denom = tp + fn
        if denom == 0:
            # Class appears in y_true but no samples? Shouldn't happen, but safe.
            recalls.append(0)
        else:
            recalls.append(tp / denom)

    # Balanced accuracy is macro recall
    return float(np.mean(recalls))

# Function used to generate JSONL predictions based on input data
# Outputs structured python list in JSONL format
def generate_results(model, dataloader, device, dataframe, tokenizer):
    model.eval()

    asp_prediction_list = []
    extracted_aspects = []
    extracted_opinions = []
    asp_combined_preds = []

    model.freeze_regressor()
    model.freeze_backbone()

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        with torch.no_grad():
            va_scores, lin_output = model(input_ids, attention_mask)
        
        batch_results_list = va_scores.cpu().tolist()
        pred_labs = model.crf_layer.decode(lin_output, mask=attention_mask.bool())

        for i in range(len(pred_labs)):
            opn_emb = []
            asp_emb = []

            current_emb = input_ids[i]
            current_pred_lab = pred_labs[i]
            
            for lab_i in range(len(current_pred_lab)):

                if (current_pred_lab[lab_i] == 2) or (current_pred_lab[lab_i] == 3) or (current_pred_lab[lab_i] == 4) or (current_pred_lab[lab_i] == 5):
                    asp_emb.append(current_emb[lab_i])
                
                elif (current_pred_lab[lab_i] == 6) or (current_pred_lab[lab_i] == 7) or (current_pred_lab[lab_i] == 8) or (current_pred_lab[lab_i] == 9):
                    opn_emb.append(current_emb[lab_i])

            opn = "none"
            asp = "none"

            if opn_emb:
                opn = tokenizer.decode(opn_emb, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            if asp_emb:
                asp = tokenizer.decode(asp_emb, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            extracted_aspects.append(asp)
            extracted_opinions.append(opn)

        asp_prediction_list.extend(batch_results_list)

    for pred in asp_prediction_list:
        score1 = f"{pred[0]:.2f}"
        score2 = f"{pred[1]:.2f}"
        asp_combined_preds.append(f"{score1}#{score2}")

    dataframe['VA'] = asp_combined_preds
    dataframe['Opinion'] = extracted_opinions
    dataframe['Aspect'] = extracted_aspects


    nested = [
        {
            "ID": group_name,
            "Triplet": group[["Aspect", "Opinion", "VA"]].to_dict(orient="records")
        }
        for group_name, group in dataframe.groupby("ID")
    ]
    
    return nested
        