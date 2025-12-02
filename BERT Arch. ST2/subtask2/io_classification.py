import json
from typing import List, Dict
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from scipy.stats import pearsonr
from tqdm import tqdm
import math
import re
import requests

from IPython.display import display, Markdown

from sklearn.utils.class_weight import compute_class_weight
import random
from sklearn.metrics import balanced_accuracy_score

DATASET = "subtask_2_eng_laptop.json"

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

    'EMB_P': -100
}

NUM_LABELS = 9

model_name = "distilbert-base-multilingual-cased" # chage your transformer model

def load_local_json_to_df(file_name):
     df = pd.read_json(file_name, orient="records", lines=True)
     return df 


class IODataset(Dataset):
    '''
    A PyTorch Dataset for BIOES Apect classification.

    - Tags every token in sentence as Singleton, Begining, Inside, End, or Ouside aspect term.
    - Tokenizes the input using a HuggingFace tokenizer.
    - Returns:
        * input_ids: token IDs, shape [max_len]
        * attention_mask: mask, shape [max_len]
        * labels: List of BIOES labels, shape [Seq_Len], String tensor

    Args:
        dataframe (pd.DataFrame): must contain "Text", "Aspect", "BIOES".
        tokenizer: HuggingFace tokenizer.
        max_len (int): max sequence length.
    '''
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.sentences = dataframe["Text"].tolist()
        self.aspects = dataframe["Aspect"].tolist()
        self.labels = dataframe["BIOES"].tolist()
        # with open('test_file.txt', 'a') as the_file:
        #     the_file.write(str(self.labels))
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
        label_ids = [LABEL_TO_NUM[tag] for tag in self.labels[idx]]

        
        label_tensor = torch.tensor(label_ids, dtype=torch.long)

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": label_tensor
        }


class TransformerIOTagger(nn.Module):
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
    def __init__(self, model_name=model_name, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.linear_classifier = nn.Linear(self.backbone.config.hidden_size, NUM_LABELS) # transform 
        self.softmax = nn.Softmax(dim=-1)
        # self.reg_head = nn.Linear(self.backbone.config.hidden_size, 2)  # Valence + Arousal


    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        token_output = outputs.last_hidden_state # entire token output
        x = self.dropout(token_output)
        # x = cls_output
        logits = self.linear_classifier(token_output)
        # probabilities = self.softmax(logits)
        return logits
        # return self.reg_head(x)

    def train_epoch(self, dataloader, optimizer, loss_fn, device):
        self.train()
        total_loss = 0
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()

            outputs = self(input_ids, attention_mask)
            outputs = outputs.view(-1, outputs.shape[-1])  # [B*seq_len, num_labels]
            labels = labels.view(-1)  
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        return total_loss / len(dataloader)

    def eval_epoch(self, dataloader, loss_fn, device):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = self(input_ids, attention_mask)

                outputs = outputs.view(-1, outputs.shape[-1])  # [B*seq_len, num_labels]
                labels = labels.view(-1)  

                loss = loss_fn(outputs, labels)
                total_loss += loss.item()
        return total_loss / len(dataloader)


# since I am using scikit-learn balanced accuracy, I need to do all model inferences sequentially, no batches 
def model_inference_bal_acc(model, dataloader, device):
    model.eval()
    accuracies = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        with torch.no_grad():
            logits = model(input_ids, attention_mask)

        preds = logits.argmax(dim=-1)  

        for index in range(preds.shape[0]):
            acc = balanced_accuracy_ignore_pads(labels[index], preds[index], -100)
            accuracies.append(acc)

    len_acc = len(accuracies)
    av_bal_acc = sum(accuracies)/ len_acc 
    return av_bal_acc 

def balanced_accuracy_ignore_pads(y_true, y_pred, ignore_label=None):
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    if ignore_label is not None:
        mask = y_true_np != ignore_label
        y_true_np = y_true_np[mask]
        y_pred_np = y_pred_np[mask]


    if len(y_true_np) == 0:
        return 0  # edge case: all padding
    return balanced_accuracy_score(y_true_np, y_pred_np)

# Main driver program -----------------------------------------------------------
def main():
    torch.cuda.empty_cache()
    # Load JSON to Pandas DF
    data_set = load_local_json_to_df(DATASET)


    all_labels = [label for seq in data_set["BIOES"] for label in seq]  # flatten list of lists

    classes = np.unique(all_labels)

    # Weighted classes for loss function. Unused 

    # weights = compute_class_weight('balanced', classes=classes, y=all_labels)
    # class_weights = torch.tensor(weights, dtype=torch.float)

    # print(class_weights)
    train_df, dev_df = train_test_split(data_set, test_size=0.15, random_state=17)

    display(Markdown(f"### train_df"))
    display(train_df.head())

    display(Markdown(f"### dev_df"))
    display(dev_df.head())

    # Tokenize Dataset 

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = IODataset(train_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    dev_dataset = IODataset(dev_df, tokenizer)
    dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=True)


    lr = 1e-5 #learning rate
    epochs = 10

    model = TransformerIOTagger().to(device)
    lr = locals().get("lr", 1e-5)
    epochs = locals().get("epochs", 5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(epochs):
        train_loss = model.train_epoch(train_loader, optimizer, loss_fn, device)
        val_loss = model.eval_epoch(dev_loader, loss_fn, device)
        print(f"model:{model_name} Epoch:{epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")


    model.eval()

    # using balanced accuracy to evaluate sequence tagging 

    train_loader_whole = DataLoader(train_dataset, batch_size= 500)
    dev_loader_whole = DataLoader(dev_dataset, batch_size =500)

    print("Train Bal acc:\n")
    print(model_inference_bal_acc(model, train_loader_whole, device))
    print("Val bal acc:")
    print(model_inference_bal_acc(model, dev_loader_whole, device))

if __name__ == "__main__":
    main()