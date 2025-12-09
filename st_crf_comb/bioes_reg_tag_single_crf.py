import json
from typing import List, Dict
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from torchcrf import CRF

from scipy.stats import pearsonr
from tqdm import tqdm
import math
import re
import requests

from IPython.display import display, Markdown

from sklearn.utils.class_weight import compute_class_weight
import random
from sklearn.metrics import balanced_accuracy_score

import ast 

# from subtask2_driver_crf import MODEL_NAME
# DATASET = "/home/tony-arant/NLP Final Project/data_labs/single_asp_opn/distilbert/distilbert_subtask_2_eng_laptop_comb.json"

REG_FLAG = 1
TAG_FLAG = 0

CLS_SIZE = 768

LABEL_TO_NUM = {
    "EMB_O": 0,

    "ASP_B": 1,
    "ASP_I": 2,
    "ASP_E": 3,
    "ASP_S": 4,

    "OPN_B": 5,
    "OPN_I": 6,
    "OPN_E": 7,
    "OPN_S": 8,

    "EMB_P": 0,        # use 0 as padding
}

NUM_LABELS = 9

DROPOUT = 0.1

def load_local_json_to_df(file_name):
     df = pd.read_json(file_name, orient="records", lines=True)
     df.reset_index(drop=True, inplace=True)
     return df 


class VA_ASP_OPN_Comb_Dataset(Dataset):
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
        self.tag_labels = dataframe["BIOES"].tolist()
        # self.asp_labels = dataframe["BIOES_asp"].tolist()
        # self.bioes_labels = dataframe["BIOES"].tolist()
        self.va_labels = dataframe[["Valence", "Arousal"]].values.astype(float)  # VA scores
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        # text = f"{self.aspects[idx]}: {self.sentences[idx]}"
        text = self.sentences[idx]
        # print(text)
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length", 
            max_length=128,
            return_tensors="pt"
        )

        # print(self.tag_labels[idx])
        # tags = ast.literal_eval(self.tag_labels[idx]) 
        # bioes_numeric = [LABEL_TO_NUM[tag] for tag in tags]
        bioes_numeric = [LABEL_TO_NUM[tag] for tag in self.tag_labels[idx]]

        # test_masking(encoded["attention_mask"][0], asp_bioes_numeric, opn_bioes_numeric)

        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "va_labels": torch.tensor(self.va_labels[idx], dtype=torch.float),  # For regression
            "bioes_labels": torch.tensor(bioes_numeric, dtype=torch.long), # For BIOES
            # "opn_labels": torch.tensor(opn_bioes_numeric, dtype=torch.long), # For BIOES
            # "idx": idx 
        }
    

class TransformerRegTagger(nn.Module):
    '''
    A BERT-based regressor for predicting Valence and Arousal scores.

    - Uses a pretrained BERT backbone to encode text.
    - Takes the [CLS] token representation as sentence-level embedding.
    - Adds a dropout layer and a linear head to output 2 values: [Valence, Arousal].
    - Includes helper methods for one training epoch and one evaluation epoch.

    Args:
        model_name (str): HuggingFace model name, default "bert-base-multilingual-cased".
        dropout (float): Dropout rate before the regression head.

    Methods:
        train_epoch(dataloader, optimizer, loss_fn, device):
            Train the model for one epoch.
            Returns average training loss.

        eval_epoch(dataloader, loss_fn, device):
            Evaluate the model for one epoch (no gradient).
            Returns average validation loss.
    '''
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.linear_classifier = nn.Linear(self.backbone.config.hidden_size, NUM_LABELS) # transform 
        self.softmax = nn.Softmax(dim=-1)
        self.crf_layer = CRF(NUM_LABELS, batch_first=True)
        # self.reg_head = nn.Linear(self.backbone.config.hidden_size, 2)  # Valence + Arousal

        self.reg_head = nn.Sequential(

            nn.Linear(CLS_SIZE, CLS_SIZE//2),
            nn.Linear(CLS_SIZE//2, 2),

            # nn.Linear(CLS_SIZE, CLS_SIZE//2),
            # nn.ELU(),
            # nn.Linear(CLS_SIZE//2, (CLS_SIZE//2)//2),
            # nn.ELU(),
            # nn.Linear((CLS_SIZE//2)//2, 2)
         )

        self.prohibit_transition_states()

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state # entire token output
        
        x = self.dropout(hidden_states)
        linear_output = self.linear_classifier(x)

        cls_output = hidden_states[:, 0]  # first token = [CLS]
        cls_output = self.dropout(cls_output)
        va_scores = self.reg_head(cls_output)

        return va_scores, linear_output

        # return self.reg_head(x)
    
    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
    
    def unfreeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def freeze_tagger(self):
        for param in self.crf_layer.parameters():
            param.requires_grad = False

    def unfreeze_tagger(self):
        for param in self.crf_layer.parameters():
            param.requires_grad = True

    def freeze_regressor(self):
        for param in self.reg_head.parameters():
            param.requires_grad = False

    def unfreeze_regressor(self):
        for param in self.reg_head.parameters():
            param.requires_grad = True




    def train_regressor_epoch(self, dataloader, optimizer, va_loss_fn, device):
        self.train()
        total_loss = 0

        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            va_labels = batch["va_labels"].to(device) 

            optimizer.zero_grad()
            
            va_scores, lin_output = self(input_ids, attention_mask)
            
            va_loss = va_loss_fn(va_scores, va_labels)
            
            va_loss.backward()
            optimizer.step()
            total_loss += va_loss.item()
        
        return total_loss / len(dataloader)
    
    def train_tagger_epoch(self, dataloader, optimizer, va_loss_fn, device):
        self.train()
        total_loss = 0

        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bioes_labels = batch["bioes_labels"].to(device) 

            optimizer.zero_grad()
            
            va_scores, lin_output = self(input_ids, attention_mask)
            
            loss = -self.crf_layer(lin_output, bioes_labels, mask=attention_mask.bool())
            loss.backward()
            optimizer.step()

            total_loss += (loss.item()) / attention_mask.sum()
        
        return total_loss
    

        # self.train()
        # total_loss = 0
        # for batch in tqdm(dataloader):
        #     input_ids = batch["input_ids"].to(device)
        #     attention_mask = batch["attention_mask"].to(device)
            
        #     if head_flag == 1:
        #         labels = batch["asp_labels"].to(device)
        #     else: 
        #         labels = batch["opn_labels"].to(device)

            
        #     optimizer.zero_grad()

        #     outputs = self(input_ids, attention_mask)
        #     # outputs = outputs.view(-1, outputs.shape[-1])  # [B*seq_len, num_labels]
        #     # labels = labels.view(-1)  
        #     # labels, mask = adjust_mask(input_ids, labels, tokenizer)
        #     # print(labels[0])
        #     # print(attention_mask[0])
        #     # print(input_ids[0])

        #     # print(len(labels[0]))
        #     # print(len(attention_mask[0]))
        #     # print(len(input_ids[0]))
            
        #     # print(outputs.shape)
        #     # while 1:
        #     #     x =1 
        #     loss = -self.crf_layer(outputs, labels, mask=attention_mask.bool())
        #     loss.backward()
        #     optimizer.step()

        #     total_loss += (loss.item()) / attention_mask.sum()
        # # return total_loss / len(dataloader)
        # return total_loss

    def eval_epoch(self, dataloader, device, va_loss_fn, flag):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                if flag == 1:
                    va_labels = batch["va_labels"].to(device)

                    va_scores, lin_output = self(input_ids, attention_mask)

                    va_loss = va_loss_fn(va_scores, va_labels)
            
                    total_batch_loss =  va_loss
                    total_loss += total_batch_loss.item()

                else: 
                    tag_labels = batch["bioes_labels"].to(device)
                    va_scores, lin_outputs = self(input_ids, attention_mask)

                # outputs = outputs.view(-1, outputs.shape[-1])  # [B*seq_len, num_labels]
                # labels = labels.view(-1)  

                    loss = -self.crf_layer(lin_outputs, tag_labels, mask=attention_mask.bool())
                    total_loss += (loss.item()) / attention_mask.sum()
        

        if flag == 1:            
            return total_loss / len(dataloader)
        else: 
            return total_loss
                    

        #         outputs = self(input_ids, attention_mask)

        #         # outputs = outputs.view(-1, outputs.shape[-1])  # [B*seq_len, num_labels]
        #         # labels = labels.view(-1)  

        #         loss = -self.crf_layer(outputs, labels, mask=attention_mask.bool())
        #         total_loss += (loss.item())/ attention_mask.sum()
        # # return total_loss / len(dataloader)
        # return total_loss 
    
    def prohibit_transition_states(self):
        with torch.no_grad():
            self.crf_layer.transitions[0, 2] = -10000.0
            self.crf_layer.transitions[0, 3] = -10000.0
            self.crf_layer.transitions[0, 6] = -10000.0
            self.crf_layer.transitions[0, 7] = -10000.0

            self.crf_layer.transitions[1, 0] = -10000.0
            self.crf_layer.transitions[1, 1] = -10000.0
            self.crf_layer.transitions[1, 4] = -10000.0
            self.crf_layer.transitions[1, 5] = -10000.0
            self.crf_layer.transitions[1, 6] = -10000.0
            self.crf_layer.transitions[1, 7] = -10000.0
            self.crf_layer.transitions[1, 8] = -10000.0

            self.crf_layer.transitions[2, 0] = -10000.0
            self.crf_layer.transitions[2, 1] = -10000.0
            self.crf_layer.transitions[2, 4] = -10000.0
            self.crf_layer.transitions[2, 5] = -10000.0
            self.crf_layer.transitions[2, 6] = -10000.0
            self.crf_layer.transitions[2, 7] = -10000.0
            self.crf_layer.transitions[2, 8] = -10000.0

            self.crf_layer.transitions[3, 2] = -10000.0
            self.crf_layer.transitions[3, 3] = -10000.0
            self.crf_layer.transitions[3, 4] = -10000.0
            self.crf_layer.transitions[3, 6] = -10000.0
            self.crf_layer.transitions[3, 7] = -10000.0

            self.crf_layer.transitions[4, 2] = -10000.0
            self.crf_layer.transitions[4, 3] = -10000.0
            self.crf_layer.transitions[4, 6] = -10000.0
            self.crf_layer.transitions[4, 7] = -10000.0

            self.crf_layer.transitions[5, 1] = -10000.0
            self.crf_layer.transitions[5, 2] = -10000.0
            self.crf_layer.transitions[5, 3] = -10000.0
            self.crf_layer.transitions[5, 4] = -10000.0
            self.crf_layer.transitions[5, 5] = -10000.0
            self.crf_layer.transitions[5, 8] = -10000.0
            self.crf_layer.transitions[5, 0] = -10000.0

            self.crf_layer.transitions[6, 0] = -10000.0
            self.crf_layer.transitions[6, 1] = -10000.0
            self.crf_layer.transitions[6, 2] = -10000.0
            self.crf_layer.transitions[6, 4] = -10000.0
            self.crf_layer.transitions[6, 5] = -10000.0
            self.crf_layer.transitions[6, 8] = -10000.0

            self.crf_layer.transitions[7, 2] = -10000.0
            self.crf_layer.transitions[7, 3] = -10000.0
            self.crf_layer.transitions[7, 6] = -10000.0
            self.crf_layer.transitions[7, 7] = -10000.0

            self.crf_layer.transitions[8, 2] = -10000.0
            self.crf_layer.transitions[8, 3] = -10000.0
            self.crf_layer.transitions[8, 6] = -10000.0
            self.crf_layer.transitions[8, 7] = -10000.0
        # self.crf_layer.transitions[0, 2] = -10000.0


# class TransformerRegressor(nn.Module):
#     def __init__(self, model_name="distilbert-base-multilingual-cased", dropout=0.1):
#             super().__init__()
#             self.backbone = AutoModel.from_pretrained(model_name)
#             self.dropout = nn.Dropout(dropout)
#             # self.linear_layer = nn.Linear(self.backbone.config.hidden_size, NUM_LABELS) # transform 
#             # self.softmax = nn.Softmax(dim=-1)
#             # self.crf_layer = CRF(NUM_LABELS, batch_first=True)
            
#             self.reg_head = nn.Sequential(

#                 nn.Linear(CLS_SIZE, CLS_SIZE//2),
#                 nn.Linear(CLS_SIZE//2, 2),

#                 # list of all regression head archs. tried in hyperparamter tuning 
#                 # TODO: Automate this process 
                
#                 # nn.Linear(CLS_SIZE, CLS_SIZE//2),
#                 # nn.Linear(CLS_SIZE//2, (CLS_SIZE//2)//2),
#                 # nn.Linear((CLS_SIZE//2)//2, 2)
                
#                 # nn.Linear(CLS_SIZE, CLS_SIZE//2),
#                 # nn.ReLU(),
#                 # nn.Linear(CLS_SIZE//2, (CLS_SIZE//2)//2),
#                 # nn.ReLU(),
#                 # nn.Linear((CLS_SIZE//2)//2, 2)

#                 # nn.Linear(CLS_SIZE, CLS_SIZE//2),
#                 # nn.LeakyReLU(0.3),
#                 # nn.Linear(CLS_SIZE//2, (CLS_SIZE//2)//2),
#                 # nn.LeakyReLU(0.3),
#                 # nn.Linear((CLS_SIZE//2)//2, 2)

#                 # nn.Linear(CLS_SIZE, CLS_SIZE//2),
#                 # nn.GELU(),
#                 # nn.Linear(CLS_SIZE//2, (CLS_SIZE//2)//2),
#                 # nn.GELU(),
#                 # nn.Linear((CLS_SIZE//2)//2, 2)

#                 # nn.Linear(CLS_SIZE, CLS_SIZE//2),
#                 # nn.ELU(),
#                 # nn.Linear(CLS_SIZE//2, 2),

#                 # nn.Linear(CLS_SIZE, CLS_SIZE//2),
#                 # nn.ELU(),
#                 # nn.Linear(CLS_SIZE//2, (CLS_SIZE//2)//2),
#                 # nn.ELU(),
#                 # nn.Linear((CLS_SIZE//2)//2, 2)
#             )

#     def forward(self, input_ids, attention_mask):
#         outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
#         hidden_states = outputs.last_hidden_state        
#         cls_output = hidden_states[:, 0]  # first token = [CLS]
#         cls_output = self.dropout(cls_output)
#         va_scores = self.reg_head(cls_output)

#         x = self.dropout(hidden_states)
#         # linear_output = self.linear_layer(x)
        
#         return va_scores
    
#     def freeze_regressor(self):
#         for parameter in self.reg_head.parameters():
#             parameter.requires_grad = False
    
#     def unfreeze_regressor(self):
#         for parameter in self.reg_head.parameters():
#             parameter.requires_grad = True
    
#     def freeze_backbone(self):
#         for parameter in self.backbone.parameters():
#             parameter.requires_grad = False
    
#     def unfreeze_backbone(self):
#         for parameter in self.backbone.parameters():
#             parameter.requires_grad = True

#     def train_regressor_epoch(self, dataloader, optimizer, va_loss_fn, device):
#         self.train()
#         total_loss = 0

#         for batch in tqdm(dataloader):
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             va_labels = batch["va_labels"].to(device) 

#             optimizer.zero_grad()
            
#             va_scores = self(input_ids, attention_mask)
            
#             va_loss = va_loss_fn(va_scores, va_labels)
            
#             va_loss.backward()
#             optimizer.step()
#             total_loss += va_loss.item()

#         return total_loss / len(dataloader)
    
# def eval_regressor_epoch(model, dataloader, va_loss_fn, device):
#     model.eval()

#     total_loss = 0

#     with torch.no_grad():
#         for batch in dataloader:
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             va_labels = batch["va_labels"].to(device)

#             va_scores = model(input_ids, attention_mask)
            
#             va_loss = va_loss_fn(va_scores, va_labels)
            
#             total_batch_loss =  va_loss
#             total_loss += total_batch_loss.item()

#     return total_loss / len(dataloader)


# since I am using scikit-learn balanced accuracy, I need to do all model inferences sequentially, no batches 
def model_inference_bal_acc(model, dataloader, device):
    model.eval()
    accuracies = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["asp_labels"]

        
        with torch.no_grad():
            unprocessed_outs = model(input_ids, attention_mask)

        preds = model.crf_layer.decode(unprocessed_outs, mask=attention_mask.bool())

        
        for index in range(len(preds)):
            acc = balanced_accuracy_ignore_pads(labels[index], preds[index], 0)
            accuracies.append(acc)

    len_acc = len(accuracies)
    av_bal_acc = sum(accuracies)/ len_acc 
    return av_bal_acc 

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

def test_masking(attn_mask, asp_num, opn_num):
    mask_l = torch.sum(attn_mask)

    asp_valid = 0
    opn_valid = 0

    for i in asp_num:
        if i != 10:
            asp_valid += 1

    for i in opn_num:
        if i != 10:
            opn_valid += 1

    if (asp_valid != mask_l) or (opn_valid != mask_l):
        print("We have a problem!")

        print(mask_l)
        print(asp_valid)
        print(opn_valid)

        print(attn_mask)
        print(opn_num)
        print(asp_num)

        while 1:
            x = 1
    
    elif (asp_valid == mask_l) and (opn_valid == mask_l):
        print("we're good")
        # while 1:
        #     x = 1

def data_t_t_split(dataset):

    if "train" not in dataset.iloc[0]["ID"]:
        dataset_len = len(dataset)
        indices = np.arange(len(dataset))

        train_end = int(0.7 * dataset_len)
        dev_end   = train_end + int(.10 * dataset_len)
        
        # slice indices
        train_idx = indices[:train_end]
        dev_idx   = indices[train_end:dev_end]
        test_idx  = indices[dev_end:]

        train_df = dataset.iloc[train_idx].copy().reset_index(drop=True)
        dev_df   = dataset.iloc[dev_idx].copy().reset_index(drop=True)
        test_df  = dataset.iloc[test_idx].copy().reset_index(drop=True)

        return train_df, dev_df, test_df
    
    # return train_idx, dev_idx, test_idx
    else: 
        train_mask = dataset['ID'].str.contains("train", case=False, na=False)
        train_df = dataset[train_mask].copy().reset_index(drop=True)

        dev_mask = dataset['ID'].str.contains("dev", case=False, na=False)
        dev_df = dataset[dev_mask].copy().reset_index(drop=True)

        test_mask = dataset['ID'].str.contains("test", case=False, na=False)
        test_df = dataset[test_mask].copy().reset_index(drop=True)

        return train_df, dev_df, test_df

def generate_preds_dev(model, dataloader, device, dataframe, tokenizer, flag):
    asp_prediction_list = []
    extracted_aspects = []
    extracted_opinions = []
    asp_combined_preds = []

    extraction_comps = []
    # model.freeze_regressor()
    # model.freeze_backbone()


    model.eval()
    model.freeze_backbone()


    # Using Regressor Model to predict VA scores
    # for batch in dataloader:
    #     input_ids = batch["input_ids"].to(device)
    #     attention_mask = batch["attention_mask"].to(device)

    #     with torch.no_grad():
    #         va_scores = reg_model(input_ids, attention_mask)

    #     batch_results_list = va_scores.cpu().tolist()

        
    #     asp_prediction_list.extend(batch_results_list)

    # for pred in asp_prediction_list:
    #     score1 = f"{pred[0]:.2f}"
    #     score2 = f"{pred[1]:.2f}"
    #     asp_combined_preds.append(f"{score1}#{score2}")

    # # Cleaning up GPU for next model 
    # del reg_model
    torch.cuda.empty_cache()
    
    # Preparing Aspect Extractor
    

    for batch in dataloader:
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        df_indices = batch["idx"] 
        
        with torch.no_grad():
            lin_output = model(input_ids, attention_mask)

        pred_labs = model.crf_layer.decode(lin_output, mask=attention_mask.bool())

        if flag == 1:
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
                asp = "NULL"

                # if opn_emb:
                #     opn = tokenizer.decode(opn_emb, skip_special_tokens=True, clean_up_tokenization_spaces=False)

                if asp_emb:
                    asp = tokenizer.decode(asp_emb, skip_special_tokens=True, clean_up_tokenization_spaces=False)


                dataset_idx = df_indices[i].item()

                act_text = dataframe.iloc[dataset_idx]["Text"]
                # print(act_text)
                act_aspect =  dataframe.iloc[dataset_idx]["Aspect"]
                
                extraction_comps.append((asp, act_aspect, act_text))

        else:
            for i in range(len(pred_labs)):
                opn_emb = []
                # asp_emb = []

                current_emb = input_ids[i]
                current_pred_lab = pred_labs[i]
                
                for lab_i in range(len(current_pred_lab)):

                    if (current_pred_lab[lab_i] == LABEL_TO_NUM["OPN_B"]) or (current_pred_lab[lab_i] == LABEL_TO_NUM["OPN_I"]) or (current_pred_lab[lab_i] == LABEL_TO_NUM["OPN_E"]) or (current_pred_lab[lab_i] == LABEL_TO_NUM["OPN_S"]):

                        opn_emb.append(current_emb[lab_i])
                    
                    # elif (current_pred_lab[lab_i] == 6) or (current_pred_lab[lab_i] == 7) or (current_pred_lab[lab_i] == 8) or (current_pred_lab[lab_i] == 9):
                    #     opn_emb.append(current_emb[lab_i])

                opn = "none"
                # asp = "NULL"

                # if opn_emb:
                #     opn = tokenizer.decode(opn_emb, skip_special_tokens=True, clean_up_tokenization_spaces=False)

                if opn_emb:
                    opn = tokenizer.decode(opn_emb, skip_special_tokens=True, clean_up_tokenization_spaces=False)


                dataset_idx = df_indices[i].item()

                act_text = dataframe.iloc[dataset_idx]["Text"]
                # print(act_text)
                opn_aspect =  dataframe.iloc[dataset_idx]["Opinion"]
                
                extraction_comps.append((opn, opn_aspect, act_text))

        return extraction_comps

    # return extraction_comps
            # extracted_opinions.append(opn)

# Function used to generate JSONL predictions based on input data
# Outputs structured python list in JSONL format
def generate_final_predictions(reg_mod_path, model_n, dataloader, device, dataframe, tokenizer, lang):
    # model.eval()

    dataframe_c = dataframe.copy()

    va_prediction_list = []
    seq_prediction_list = []

    extracted_aspects = []
    extracted_opinions = []
    va_combined_preds = []

    # model.freeze_regressor()
    # model.freeze_backbone()

    model = TransformerRegTagger(model_n).to(device)
    model.load_state_dict(torch.load(reg_mod_path))

    model.eval()
    model.freeze_backbone()
    model.freeze_regressor()
    model.freeze_tagger()


    # Using Regressor Model to predict VA scores
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            va_scores, lin_output = model(input_ids, attention_mask)

        pred_seq = model.crf_layer.decode(lin_output, mask=attention_mask.bool())
        va_scores_list= va_scores.cpu().tolist()
        
        

        
        va_prediction_list.extend(va_scores_list)
        seq_prediction_list = pred_seq

    # for pred in va_prediction_list:
    #     score1 = f"{pred[0]:.2f}"
    #     score2 = f"{pred[1]:.2f}"
    #     va_combined_preds.append(f"{score1}#{score2}")

        # print(len(input_ids))
        # print(len(seq_prediction_list))


        for i in range(len(seq_prediction_list)):
            opn_emb = []
            asp_emb = []

            current_emb = input_ids[i]
            current_pred_lab = seq_prediction_list[i]
            
            # print(len(current_emb))
            # print(len(current_pred_lab), end="\n\n") 
            for lab_i in range(len(current_pred_lab)):

                if (current_pred_lab[lab_i] == LABEL_TO_NUM["ASP_B"]) or (current_pred_lab[lab_i] == LABEL_TO_NUM["ASP_I"]) or (current_pred_lab[lab_i] == LABEL_TO_NUM["ASP_E"]) or (current_pred_lab[lab_i] == LABEL_TO_NUM["ASP_S"]):

                    asp_emb.append(current_emb[lab_i])
                
                elif (current_pred_lab[lab_i] == LABEL_TO_NUM["OPN_B"]) or (current_pred_lab[lab_i] == LABEL_TO_NUM["OPN_I"]) or (current_pred_lab[lab_i] == LABEL_TO_NUM["OPN_E"]) or (current_pred_lab[lab_i] == LABEL_TO_NUM["OPN_S"]):
                    opn_emb.append(current_emb[lab_i])

            reconstructed_opinion = "none"
            reconstructed_aspect = "none"

            # if opn_emb:
            #     opn = tokenizer.decode(opn_emb, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            if asp_emb:
                reconstructed_aspect = tokenizer.decode(asp_emb, skip_special_tokens=True, clean_up_tokenization_spaces=False)

                if lang == "rus":
                    reconstructed_aspect = re.sub(r' - ', '-', reconstructed_aspect)
                    reconstructed_aspect   = re.sub(r'([0-9]) \+', r'\1+', reconstructed_aspect)
                    reconstructed_aspect   = re.sub(r'([0-9]) %',  r'\1%', reconstructed_aspect)
                    reconstructed_aspect = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', reconstructed_aspect)

                if lang == "tat":
                    reconstructed_aspect = re.sub(r' - ', '-', reconstructed_aspect)
                    reconstructed_aspect = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', reconstructed_aspect)
                    reconstructed_aspect   = re.sub(r'([0-9]) \+', r'\1+', reconstructed_aspect)
                    reconstructed_aspect   = re.sub(r'([0-9]) %',  r'\1%', reconstructed_aspect)

                if lang == "ukr":
                    reconstructed_aspect = re.sub(r' - ', '-', reconstructed_aspect)
                    reconstructed_aspect = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', reconstructed_aspect)
                    reconstructed_aspect = re.sub(r' \' ', '\'', reconstructed_aspect)
                    reconstructed_aspect   = re.sub(r'([0-9]) \+', r'\1+', reconstructed_aspect)
                    reconstructed_aspect   = re.sub(r'([0-9]) %',  r'\1%', reconstructed_aspect)

                if lang == "zho":
                    reconstructed_aspect = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', reconstructed_aspect)
                    reconstructed_aspect = re.sub(r"\s+", "", reconstructed_aspect)


            if opn_emb:
                reconstructed_opinion = tokenizer.decode(opn_emb, skip_special_tokens=True, clean_up_tokenization_spaces=False)

                if lang == "rus":
                    reconstructed_opinion = re.sub(r' - ', '-', reconstructed_opinion)
                    reconstructed_opinion  = re.sub(r'([0-9]) \+', r'\1+', reconstructed_opinion)
                    reconstructed_opinion  = re.sub(r'([0-9]) %',  r'\1%', reconstructed_opinion)
                    reconstructed_opinion = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', reconstructed_opinion)
                
                if lang == "tat":
                    reconstructed_opinion = re.sub(r' - ', '-', reconstructed_opinion)
                    reconstructed_opinion = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', reconstructed_opinion)
                    reconstructed_opinion  = re.sub(r'([0-9]) \+', r'\1+', reconstructed_opinion)
                    reconstructed_opinion  = re.sub(r'([0-9]) %',  r'\1%', reconstructed_opinion)

                if lang == "ukr":
                    reconstructed_opinion = re.sub(r' - ', '-', reconstructed_opinion)
                    reconstructed_opinion = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', reconstructed_opinion)
                    reconstructed_opinion = re.sub(r' \' ', '\'', reconstructed_opinion)
                    reconstructed_opinion  = re.sub(r'([0-9]) \+', r'\1+', reconstructed_opinion)
                    reconstructed_opinion  = re.sub(r'([0-9]) %',  r'\1%', reconstructed_opinion)

            
                if lang == "zho":
                    reconstructed_opinion = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', reconstructed_opinion)
                    reconstructed_opinion = re.sub(r"\s+", "", reconstructed_opinion)

                    
                
            extracted_aspects.append(reconstructed_aspect)
            extracted_opinions.append(reconstructed_opinion)

    # print(extracted_aspects)
    # print(extracted_opinions)

    for pred in va_prediction_list:
        score1 = f"{pred[0]:.2f}"
        score2 = f"{pred[1]:.2f}"
        va_combined_preds.append(f"{score1}#{score2}")

    
    # Cleaning up GPU for next model 

    # # Preparing Aspect Extractor
    # asp_tag_model = TransformerTagger().to(device)
    # asp_tag_model.load_state_dict(torch.load(asp_mod_path))

    # asp_tag_model.eval()
    # asp_tag_model.freeze_backbone()
    # asp_tag_model.freeze_tagger()

    # for batch in dataloader:
    #     input_ids = batch["input_ids"].to(device)
    #     attention_mask = batch["attention_mask"].to(device)
        
    #     with torch.no_grad():
    #         lin_output = asp_tag_model(input_ids, attention_mask)

    #     # pred_labs = asp_tag_model.softmax(lin_output)
    #     # pred_labs = torch.argmax(pred_labs, dim=-1).cpu().tolist()

    #     pred_labs = asp_tag_model.crf_layer.decode(lin_output, mask=attention_mask.bool())

    #     for i in range(len(pred_labs)):
    #         # opn_emb = []
    #         asp_emb = []

    #         current_emb = input_ids[i]
    #         current_pred_lab = pred_labs[i]
            
    #         for lab_i in range(len(current_pred_lab)):

    #             if (current_pred_lab[lab_i] == LABEL_TO_NUM["ASP_B"]) or (current_pred_lab[lab_i] == LABEL_TO_NUM["ASP_I"]) or (current_pred_lab[lab_i] == LABEL_TO_NUM["ASP_E"]) or (current_pred_lab[lab_i] == LABEL_TO_NUM["ASP_S"]):

    #                 asp_emb.append(current_emb[lab_i])
                
    #             # elif (current_pred_lab[lab_i] == 6) or (current_pred_lab[lab_i] == 7) or (current_pred_lab[lab_i] == 8) or (current_pred_lab[lab_i] == 9):
    #             #     opn_emb.append(current_emb[lab_i])

    #         # opn = "none"
    #         asp = "none"

    #         # if opn_emb:
    #         #     opn = tokenizer.decode(opn_emb, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    #         if asp_emb:
    #             asp = tokenizer.decode(asp_emb, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
    #         extracted_aspects.append(asp)
    #         # extracted_opinions.append(opn)

    # del asp_tag_model
    # torch.cuda.empty_cache()

    # # Preparing Opinion Extractor

    # opn_tag_model = TransformerTagger().to(device)
    # opn_tag_model.load_state_dict(torch.load(opn_mod_path))

    # opn_tag_model.eval()
    # opn_tag_model.freeze_backbone()
    # opn_tag_model.freeze_tagger()

    # for batch in dataloader:
    #     input_ids = batch["input_ids"].to(device)
    #     attention_mask = batch["attention_mask"].to(device)
        
    #     with torch.no_grad():
    #         lin_output = opn_tag_model(input_ids, attention_mask)

    #     pred_labs = opn_tag_model.crf_layer.decode(lin_output, mask=attention_mask.bool())


    #     for i in range(len(pred_labs)):
    #         opn_emb = []
    #         # asp_emb = []

    #         current_emb = input_ids[i]
    #         current_pred_lab = pred_labs[i]
            
    #         for lab_i in range(len(current_pred_lab)):

    #             # if (current_pred_lab[lab_i] == 2) or (current_pred_lab[lab_i] == 3) or (current_pred_lab[lab_i] == 4) or (current_pred_lab[lab_i] == 5):
    #             #     asp_emb.append(current_emb[lab_i])
                
    #             if (current_pred_lab[lab_i] == LABEL_TO_NUM["OPN_B"]) or (current_pred_lab[lab_i] == LABEL_TO_NUM["OPN_I"]) or (current_pred_lab[lab_i] == LABEL_TO_NUM["OPN_E"]) or (current_pred_lab[lab_i] == LABEL_TO_NUM["OPN_S"]):
    #                 opn_emb.append(current_emb[lab_i])

    #         opn = "none"
    #         # asp = "none"

    #         if opn_emb:
    #             opn = tokenizer.decode(opn_emb, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    #         # if asp_emb:
    #         #     asp = tokenizer.decode(asp_emb, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
    #         # extracted_aspects.append(asp)
    #         extracted_opinions.append(opn)

    dataframe_c['VA'] = va_combined_preds
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
         
# Main driver program -----------------------------------------------------------
def main():
    torch.cuda.empty_cache()

    model = TransformerTagger(MODEL_NAME, DROPOUT)


    # Load JSON to Pandas DF
    data_set = load_local_json_to_df(DATASET)

    # Make Train-Dev-Test split from data file
    train_df, dev_df, test_df = data_t_t_split(data_set)

    display(Markdown(f"### train_df"))
    display(train_df.head())

    display(Markdown(f"### dev_df"))
    display(dev_df.head())

    display(Markdown("### test_df"))
    display(test_df)


    # all_labels = [label for seq in data_set["BIOES"] for label in seq]  # flatten list of lists

    # classes = np.unique(all_labels)

    # # Weighted classes for loss function. Unused 

    # # weights = compute_class_weight('balanced', classes=classes, y=all_labels)
    # # class_weights = torch.tensor(weights, dtype=torch.float)

    # # print(class_weights)
    train_df, dev_df = train_test_split(data_set, test_size=0.15, random_state=17)

    # display(Markdown(f"### train_df"))
    # display(train_df.head())

    # display(Markdown(f"### dev_df"))
    # display(dev_df.head())

    # # Tokenize Dataset 

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = VA_ASP_OPN_Split_Dataset(train_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    dev_dataset = VA_ASP_OPN_Split_Dataset(dev_df, tokenizer)
    dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=True)


    lr = 1e-4 #learning rate
    epochs = 4

    model = model.to(device)
    lr = locals().get("lr", 1e-5)
    epochs = locals().get("epochs", 5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # model.freeze_backbone()

    for epoch in range(epochs):
        train_loss = model.train_epoch_tagger(train_loader, optimizer, device, 1)
        val_loss = model.eval_epoch(dev_loader, device, 1)
        print(f"model:{MODEL_NAME} Epoch:{epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")


    preds = generate_preds_dev(model, dev_loader, device, dev_df, tokenizer, 1)

    # print(preds)

    with open("extracted_asp2.txt", "w") as f:
        f.write(f"extracted\tactual\ttext\n")
        for p in preds:
            f.write(f"{p[0]} | \t")
            f.write(f"{p[1]} | \t")
            f.write(f"{p[2]}\n")



    # model.eval()

    # # using balanced accuracy to evaluate sequence tagging 

    # train_loader_whole = DataLoader(train_dataset, batch_size= 500)
    # dev_loader_whole = DataLoader(dev_dataset, batch_size =500)

    # print("Train Bal acc:\n")
    # print(model_inference_bal_acc(model, train_loader_whole, device))
    # print("Val bal acc:")
    # print(model_inference_bal_acc(model, dev_loader_whole, device))

if __name__ == "__main__":
    main()