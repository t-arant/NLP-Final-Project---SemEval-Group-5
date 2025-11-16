import json
from typing import List, Dict
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from IPython.display import display, Markdown

from sklearn.utils.class_weight import compute_class_weight

from bioes_regression_combined import (
    CombinedModel, 
    CombinedDataset, 
    train_combined_epoch,
    eval_combined_epoch, 
    model_inference_combined
)
from bioes_classification import (
    load_local_json_to_df, 
    model_name,
    DATASET
)

def main():
    # Load JSON to Pandas DF
    data_set = load_local_json_to_df(DATASET)

    all_labels = [label for seq in data_set["BIOES"] for label in seq]  # flatten

    classes = np.unique(all_labels)
    weights = compute_class_weight('balanced', classes=classes, y=all_labels)
    class_weights = torch.tensor(weights, dtype=torch.float)

    train_df, dev_df = train_test_split(data_set, test_size=0.15, random_state=17)

    display(Markdown(f"### train_df"))
    display(train_df.head())

    display(Markdown(f"### dev_df"))
    display(dev_df.head())

    # Tokenize Dataset 
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = CombinedDataset(train_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    dev_dataset = CombinedDataset(dev_df, tokenizer)
    dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=True)

    lr = 1e-5 #learning rate
    epochs = 8

    model = CombinedModel().to(device)
    lr = locals().get("lr", 1e-5)
    epochs = locals().get("epochs", 5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # two loss functions instead of one
    bioes_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    va_loss_fn = nn.MSELoss()

    # loop doesn't use functions inside the model class
    for epoch in range(epochs):
        train_loss = train_combined_epoch(model, train_loader, optimizer, bioes_loss_fn, va_loss_fn, device)
    
        # for validation loss, we can create a similar eval function
        val_loss = eval_combined_epoch(model, dev_loader, bioes_loss_fn, va_loss_fn, device)
    
        print(f"model:{model_name} Epoch:{epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")

    model.eval()

    # using balanced accuracy to evaluate sequence tagging 
    train_loader_whole = DataLoader(train_dataset, batch_size=500)
    dev_loader_whole = DataLoader(dev_dataset, batch_size=500)

    print("Train Bal acc:\n")
    print(model_inference_combined(model, train_loader_whole))
    print("Val bal acc:")
    print(model_inference_combined(model, dev_loader_whole))
    
if __name__ == "__main__":
    main()