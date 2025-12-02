# This file contains the code for defining, training, and evaluating a BERT-based model for subtask 1 

import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel
from transformers import AutoTokenizer
from tqdm import tqdm

import torch

import numpy as np
# from torchcrf import CRF
from root_mse import rmse

CLS_SIZE = 768

NUM_LABELS = 5


LABEL_TO_NUM = {
    "EMB_O": 0,

    "ASP_B": 1,
    "ASP_I": 2,
    "ASP_E": 3,
    "ASP_S": 4,

    "OPN_B": 1,
    "OPN_I": 2,
    "OPN_E": 3,
    "OPN_S": 4,

    "EMB_P": 10,        # use 10 as padding
}

class VA_ASP_OPN_Split_Dataset(Dataset):
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
        self.opn_labels = dataframe["BIOES_opn"].tolist()
        self.asp_labels = dataframe["BIOES_asp"].tolist()
        # self.bioes_labels = dataframe["BIOES"].tolist()
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

        asp_bioes_numeric = [LABEL_TO_NUM[tag] for tag in self.asp_labels[idx]]
        opn_bioes_numeric = [LABEL_TO_NUM[tag] for tag in self.opn_labels[idx]]

        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "va_labels": torch.tensor(self.va_labels[idx], dtype=torch.float),  # For regression
            "asp_labels": torch.tensor(asp_bioes_numeric, dtype=torch.long), # For BIOES
            "opn_labels": torch.tensor(opn_bioes_numeric, dtype=torch.long), # For BIOES
        }

class BERT_Regression_Model(nn.Module):
    def __init__(self, model_name="bert-base-cased", dropout=0.1):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(model_name)
            self.dropout = nn.Dropout(dropout)
            # self.linear_layer = nn.Linear(self.backbone.config.hidden_size, NUM_LABELS) # transform 
            # self.softmax = nn.Softmax(dim=-1)
            # self.crf_layer = CRF(NUM_LABELS, batch_first=True)
            
            self.reg_head = nn.Sequential(

                nn.Linear(CLS_SIZE, CLS_SIZE//2),
                nn.Linear(CLS_SIZE//2, 2),

                # list of all regression head archs. tried in hyperparamter tuning 
                # TODO: Automate this process 
                
                # nn.Linear(CLS_SIZE, CLS_SIZE//2),
                # nn.Linear(CLS_SIZE//2, (CLS_SIZE//2)//2),
                # nn.Linear((CLS_SIZE//2)//2, 2)
                
                # nn.Linear(CLS_SIZE, CLS_SIZE//2),
                # nn.ReLU(),
                # nn.Linear(CLS_SIZE//2, (CLS_SIZE//2)//2),
                # nn.ReLU(),
                # nn.Linear((CLS_SIZE//2)//2, 2)

                # nn.Linear(CLS_SIZE, CLS_SIZE//2),
                # nn.LeakyReLU(0.3),
                # nn.Linear(CLS_SIZE//2, (CLS_SIZE//2)//2),
                # nn.LeakyReLU(0.3),
                # nn.Linear((CLS_SIZE//2)//2, 2)

                # nn.Linear(CLS_SIZE, CLS_SIZE//2),
                # nn.GELU(),
                # nn.Linear(CLS_SIZE//2, (CLS_SIZE//2)//2),
                # nn.GELU(),
                # nn.Linear((CLS_SIZE//2)//2, 2)

                # nn.Linear(CLS_SIZE, CLS_SIZE//2),
                # nn.ELU(),
                # nn.Linear(CLS_SIZE//2, 2),

                # nn.Linear(CLS_SIZE, CLS_SIZE//2),
                # nn.ELU(),
                # nn.Linear(CLS_SIZE//2, (CLS_SIZE//2)//2),
                # nn.ELU(),
                # nn.Linear((CLS_SIZE//2)//2, 2)
            )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state        
        cls_output = hidden_states[:, 0]  # first token = [CLS]
        cls_output = self.dropout(cls_output)
        va_scores = self.reg_head(cls_output)

        x = self.dropout(hidden_states)
        # linear_output = self.linear_layer(x)
        
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

class BERT_Tag_Model(nn.Module):
    '''
    A BERT-based regressor for predicting Valence and Arousal scores. 

    - Uses a pretrained BERT backbone to contextualize text embeddings.
    - Takes the [CLS] token representation as sentence-level embedding.
    - Runs [CLS] token through dropout layer and regression head
    - DOES NOT INCLUDE helper methods for one training epoch and one evaluation epoch. 

    Args:
        model_name (str): HuggingFace model name, default "bert-base-multilingual-cased".
        dropout (float): Dropout rate before the regression head.
    '''
    def __init__(self, model_name="bert-base-cased", dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.linear_layer = nn.Linear(self.backbone.config.hidden_size, NUM_LABELS) # transform 
        self.softmax = nn.Softmax(dim=-1)
        # self.crf_layer = CRF(NUM_LABELS, batch_first=True)
        
        
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state        
        # cls_output = hidden_states[:, 0]  # first token = [CLS]
        # cls_output = self.dropout(cls_output)
        # va_scores = self.reg_head(cls_output)

        x = self.dropout(hidden_states)
        linear_output = self.linear_layer(x)
        
        return linear_output
    
    # def freeze_regressor(self):
    #     for parameter in self.reg_head.parameters():
    #         parameter.requires_grad = False
    
    # def unfreeze_regressor(self):
    #     for parameter in self.reg_head.parameters():
    #         parameter.requires_grad = True

    def freeze_tagger(self):
        for parameter in self.linear_layer.parameters():
            parameter.requires_grad = False
    
    def unfreeze_tagger(self):
        for parameter in self.linear_layer.parameters():
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

def train_tagger_epoch(model, dataloader, optimizer, tag_loss_fn, device, extract_flag):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        if extract_flag == 1: 
            tag_labels = batch["asp_labels"].to(device)

        elif extract_flag == 2: 
            tag_labels = batch["opn_labels"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)

        outputs = outputs.view(-1, outputs.shape[-1])  # [B*seq_len, num_labels]
        tag_labels = tag_labels.view(-1) 
        attention_mask = attention_mask.view(-1)  
        
        active_indices = attention_mask == 1
        active_outputs = outputs[active_indices]        # [N_active, num_labels]
        active_labels = tag_labels[active_indices]      # [N_active]

        # print(active_labels.unique())
        # print(active_outputs.unique())
        # while 1: 
        #     x= 1
        # Compute loss
        loss = tag_loss_fn(active_outputs, active_labels)


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

            va_scores = model(input_ids, attention_mask)
            
            va_loss = va_loss_fn(va_scores, va_labels)
            
            total_batch_loss =  va_loss
            total_loss += total_batch_loss.item()
    
    return total_loss / len(dataloader)

def eval_tagger_epoch(model, dataloader, tag_loss_fn, device, extract_flag):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                if extract_flag == 1: 
                    tag_labels = batch["asp_labels"].to(device)

                elif extract_flag == 2: 
                    tag_labels = batch["opn_labels"].to(device)

                outputs = model(input_ids, attention_mask)

                outputs = outputs.view(-1, outputs.shape[-1])  # [B*seq_len, num_labels]
                tag_labels = tag_labels.view(-1) 
                attention_mask = attention_mask.view(-1)  
                
                active_indices = attention_mask == 1
                active_outputs = outputs[active_indices]        # [N_active, num_labels]
                active_labels = tag_labels[active_indices]      # [N_active]

                # Compute loss
                loss = tag_loss_fn(active_outputs, active_labels)


                total_loss += loss.item()
        return total_loss / len(dataloader)

# Function for evaluating model after training
# Returns average of error for whole dataset provided 
def model_inference_reg(model, reg_loss_fn, dataloader, device):
    model.eval()
    va_errors = []
    # tag_errors = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        va_labels = batch["va_labels"].to(device)
        # tag_labels = batch["bioes_labels"].to(device)
        

        with torch.no_grad():
            va_scores  = model(input_ids, attention_mask)
        
        # VA regression error (new)
        va_error = reg_loss_fn(va_scores, va_labels.to(device))
        va_errors.append(va_error.item())

        # tag_output = tag_output.view(-1, tag_output.shape[-1])  # [B*seq_len, num_labels]
        # tag_labels = tag_labels.view(-1) 
        # attention_mask = attention_mask.view(-1)  
        
        # active_indices = attention_mask == 1
        # active_outputs = tag_output[active_indices]        # [N_active, num_labels]
        # active_labels = tag_labels[active_indices]      # [N_active]

        # # Compute loss
        # t_loss = tag_loss_fn(active_outputs, active_labels)
        # tag_errors.append(t_loss.item())

    avg_va_error = sum(va_errors) / len(va_errors)
    # avg_tag_error = sum(tag_errors) / len(tag_errors)
    
    return avg_va_error

def model_inference_tag(model, tag_loss_fn, dataloader, device, flag):
    model.eval()
    va_errors = []
    tag_errors = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        # va_labels = batch["va_labels"].to(device)
        # tag_labels = batch["bioes_labels"].to(device)

        if flag == 1: 
            tag_labels = batch["asp_labels"].to(device)

        elif flag == 2: 
            tag_labels = batch["opn_labels"].to(device)
        

        with torch.no_grad():
            tag_output  = model(input_ids, attention_mask)
        
        # VA regression error (new)
        # va_error = reg_loss_fn(va_scores, va_labels.to(device))
        # va_errors.append(va_error.item())

        tag_output = tag_output.view(-1, tag_output.shape[-1])  # [B*seq_len, num_labels]
        tag_labels = tag_labels.view(-1) 
        attention_mask = attention_mask.view(-1)  
        
        active_indices = attention_mask == 1
        active_outputs = tag_output[active_indices]        # [N_active, num_labels]
        active_labels = tag_labels[active_indices]      # [N_active]

        # Compute loss
        t_loss = tag_loss_fn(active_outputs, active_labels)
        tag_errors.append(t_loss.item())

    # avg_va_error = sum(va_errors) / len(va_errors)
    avg_tag_error = sum(tag_errors) / len(tag_errors)
    
    return avg_tag_error

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
def generate_results(reg_mod_path, asp_mod_path, opn_mod_path, dataloader, device, dataframe, tokenizer):
    # model.eval()

    dataframe_c = dataframe.copy()

    asp_prediction_list = []
    extracted_aspects = []
    extracted_opinions = []
    asp_combined_preds = []

    # model.freeze_regressor()
    # model.freeze_backbone()

    reg_model = BERT_Regression_Model().to(device)
    reg_model.load_state_dict(torch.load(reg_mod_path))

    reg_model.eval()
    reg_model.freeze_backbone()
    reg_model.freeze_regressor()

    # Using Regressor Model to predict VA scores
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            va_scores = reg_model(input_ids, attention_mask)

        batch_results_list = va_scores.cpu().tolist()

        
        asp_prediction_list.extend(batch_results_list)

    for pred in asp_prediction_list:
        score1 = f"{pred[0]:.2f}"
        score2 = f"{pred[1]:.2f}"
        asp_combined_preds.append(f"{score1}#{score2}")

    # Cleaning up GPU for next model 
    del reg_model
    torch.cuda.empty_cache()
    
    # Preparing Aspect Extractor
    asp_tag_model = BERT_Tag_Model().to(device)
    asp_tag_model.load_state_dict(torch.load(asp_mod_path))

    asp_tag_model.eval()
    asp_tag_model.freeze_backbone()
    asp_tag_model.freeze_tagger()

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        with torch.no_grad():
            lin_output = asp_tag_model(input_ids, attention_mask)

        pred_labs = asp_tag_model.softmax(lin_output)
        pred_labs = torch.argmax(pred_labs, dim=-1).cpu().tolist()

        for i in range(len(pred_labs)):
            # opn_emb = []
            asp_emb = []

            current_emb = input_ids[i]
            current_pred_lab = pred_labs[i]
            
            for lab_i in range(len(current_pred_lab)):

                if (current_pred_lab[lab_i] == LABEL_TO_NUM["ASP_B"]) or (current_pred_lab[lab_i] == LABEL_TO_NUM["ASP_I"]) or (current_pred_lab[lab_i] == LABEL_TO_NUM["ASP_E"]) or (current_pred_lab[lab_i] == LABEL_TO_NUM["ASP_S"]):

                    asp_emb.append(current_emb[lab_i])
                
                # elif (current_pred_lab[lab_i] == 6) or (current_pred_lab[lab_i] == 7) or (current_pred_lab[lab_i] == 8) or (current_pred_lab[lab_i] == 9):
                #     opn_emb.append(current_emb[lab_i])

            # opn = "none"
            asp = "none"

            # if opn_emb:
            #     opn = tokenizer.decode(opn_emb, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            if asp_emb:
                asp = tokenizer.decode(asp_emb, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            extracted_aspects.append(asp)
            # extracted_opinions.append(opn)

    del asp_tag_model
    torch.cuda.empty_cache()

    # Preparing Opinion Extractor

    opn_tag_model = BERT_Tag_Model().to(device)
    opn_tag_model.load_state_dict(torch.load(opn_mod_path))

    opn_tag_model.eval()
    opn_tag_model.freeze_backbone()
    opn_tag_model.freeze_tagger()

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        with torch.no_grad():
            lin_output = opn_tag_model(input_ids, attention_mask)

        pred_labs = opn_tag_model.softmax(lin_output)
        pred_labs = torch.argmax(pred_labs, dim=-1).cpu().tolist()

        for i in range(len(pred_labs)):
            opn_emb = []
            # asp_emb = []

            current_emb = input_ids[i]
            current_pred_lab = pred_labs[i]
            
            for lab_i in range(len(current_pred_lab)):

                # if (current_pred_lab[lab_i] == 2) or (current_pred_lab[lab_i] == 3) or (current_pred_lab[lab_i] == 4) or (current_pred_lab[lab_i] == 5):
                #     asp_emb.append(current_emb[lab_i])
                
                if (current_pred_lab[lab_i] == LABEL_TO_NUM["OPN_B"]) or (current_pred_lab[lab_i] == LABEL_TO_NUM["OPN_I"]) or (current_pred_lab[lab_i] == LABEL_TO_NUM["OPN_E"]) or (current_pred_lab[lab_i] == LABEL_TO_NUM["OPN_S"]):
                    opn_emb.append(current_emb[lab_i])

            opn = "none"
            # asp = "none"

            if opn_emb:
                opn = tokenizer.decode(opn_emb, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            # if asp_emb:
            #     asp = tokenizer.decode(asp_emb, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            # extracted_aspects.append(asp)
            extracted_opinions.append(opn)
    # ---------------------------------------------------------------------------------------------------------

    # for batch in dataloader:
    #     input_ids = batch["input_ids"].to(device)
    #     attention_mask = batch["attention_mask"].to(device)
        
    #     with torch.no_grad():
    #         va_scores, lin_output = model(input_ids, attention_mask)
        
    #     batch_results_list = va_scores.cpu().tolist()
    #     pred_labs = model.softmax(lin_output)
    #     pred_labs = torch.argmax(pred_labs, dim=-1).cpu().tolist()

    #     for i in range(len(pred_labs)):
    #         opn_emb = []
    #         asp_emb = []

    #         current_emb = input_ids[i]
    #         current_pred_lab = pred_labs[i]
            
    #         for lab_i in range(len(current_pred_lab)):

    #             if (current_pred_lab[lab_i] == 2) or (current_pred_lab[lab_i] == 3) or (current_pred_lab[lab_i] == 4) or (current_pred_lab[lab_i] == 5):
    #                 asp_emb.append(current_emb[lab_i])
                
    #             elif (current_pred_lab[lab_i] == 6) or (current_pred_lab[lab_i] == 7) or (current_pred_lab[lab_i] == 8) or (current_pred_lab[lab_i] == 9):
    #                 opn_emb.append(current_emb[lab_i])

    #         opn = "none"
    #         asp = "none"

    #         if opn_emb:
    #             opn = tokenizer.decode(opn_emb, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    #         if asp_emb:
    #             asp = tokenizer.decode(asp_emb, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
    #         extracted_aspects.append(asp)
    #         extracted_opinions.append(opn)

    #     asp_prediction_list.extend(batch_results_list)

    # for pred in asp_prediction_list:
    #     score1 = f"{pred[0]:.2f}"
    #     score2 = f"{pred[1]:.2f}"
    #     asp_combined_preds.append(f"{score1}#{score2}")

    dataframe_c['VA'] = asp_combined_preds
    dataframe_c['Opinion'] = extracted_opinions
    dataframe_c['Aspect'] = extracted_aspects


    nested = [
        {
            "ID": group_name,
            "Triplet": group[["Aspect", "Opinion", "VA"]].to_dict(orient="records")
        }
        for group_name, group in dataframe_c.groupby("ID")
    ]
    
    return nested
        