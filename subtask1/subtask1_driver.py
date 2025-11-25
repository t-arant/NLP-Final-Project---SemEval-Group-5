# Driver program for subtask 1
# Uses command line arguments to select what mode of use 1 - Training Model, 0 - Model is frozen and used to generate predictions JSON

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

import sys 

from subtask1_model import (
    RegressorModel, 
    VADataset, 
    eval_regressor_epoch,
    train_regressor_epoch,
    model_inference, 
    generate_results
)
from data_loader import (
    generate_gold_file,
    load_local_json_to_df, 
    model_name,
    DATASET
)

from root_mse import rmse

# Takes in DataFrame, splits based on ID label of train, dev, or test
# returns split data in 3 DataFrames
def data_t_t_split(dataset):
    train_mask = dataset['ID'].str.contains("train", case=False, na=False)
    train_df = dataset[train_mask].copy().reset_index(drop=True)

    dev_mask = dataset['ID'].str.contains("dev", case=False, na=False)
    dev_df = dataset[dev_mask].copy().reset_index(drop=True)

    test_mask = dataset['ID'].str.contains("test", case=False, na=False)
    test_df = dataset[test_mask].copy().reset_index(drop=True)

    return train_df, dev_df, test_df


def main():
    train = int(sys.argv[1])

    # Load JSON to Pandas DF
    data_set = load_local_json_to_df(DATASET)

    # Make Train-Dev-Test split from data file
    train_df, dev_df, test_df = data_t_t_split(data_set)

    display(Markdown(f"### train_df"))
    display(train_df.head())

    display(Markdown(f"### dev_df"))
    display(dev_df.head())

    display(Markdown("Train Loc 1"))
    display(train_df.iloc[3])

    # Define Tokenizer 
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

    # Define Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # runs if model is in training mode
    if train == 1:
        
        # Create tokenized train and dev data, and encapsulate into DataLoader 
        train_dataset = VADataset(train_df, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        dev_dataset = VADataset(dev_df, tokenizer)
        dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False)

        lr = 1e-5 #learning rate

        # sets desired epochs for training 
        # TODO: fix and automate
        epochs_reg = 5
        
        # sets index of output loss graph data
        # TODO: fix and automate 
        trial = -100
        
        # Send model to device and define learning rate, epochs, optimizer function, and loss funct. 
        model = RegressorModel().to(device)
        lr = locals().get("lr", 1e-5)
        r_epochs = locals().get("epochs_reg", 5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        va_loss_fn = rmse()

        # model freezing based on desired arch. 
        # TODO: fix this so that it can be automated with desired regression head arch. 
        
        # model.freeze_backbone()

        print("\n\nTraining Regressor\n\n")
        train_losses = []

        for epoch in range(r_epochs):
            train_loss = train_regressor_epoch(model, train_loader, optimizer, va_loss_fn, device)
        
            # for validation loss, we can create a similar eval function
            val_loss = eval_regressor_epoch(model, dev_loader, va_loss_fn, device)
        
            print(f"model:{model_name} Epoch:{epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")
            train_losses.append((epoch, train_loss, val_loss))
        
        model.eval()

        train_loader_whole = DataLoader(train_dataset, batch_size=500)
        dev_loader_whole = DataLoader(dev_dataset, batch_size=500)

        print("\nTrain Root mse:")
        final_train_loss = model_inference(model, va_loss_fn, train_loader_whole, device)
        print(final_train_loss)
        print("Val Root mse:")
        final_val_loss = model_inference(model, va_loss_fn, dev_loader_whole, device)
        print(final_val_loss)

        train_losses.append(("eval", final_train_loss, final_val_loss))

        # place logged loss data into a DF, then write to CSV
        loss_df = pd.DataFrame(train_losses, columns=['Epoch', 'Train Loss', 'Validation Loss'])
        loss_df.to_csv(f"loss_graphs/subtask1_{trial}.csv", index=False)

        # added this in to save state dictionary of final model, so we can load this and not train a new model for final eval
        # TODO: only save model if training final model arch. 
        torch.save(model.state_dict(), "english_laptop_final.pth")

    # Runs if model is in test mode
    # Returns a JSONL of VA predictions from text
    elif train == 0: 
        # load state dictionary and sent to GPU
        model = RegressorModel()
        model.load_state_dict(torch.load("english_laptop_final.pth"))
        model = model.to(device)

        # Tokenize test datasets, and load into DataLoader
        test_dataset = VADataset(test_df, tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        output = generate_results(model, test_loader, device, test_df)
        
        # write predictions to orgainizer specified formate file
        with open("subtask_1_eng_laptop_preds.json", "w", encoding="utf-8") as f:
            for out in output:
               f.write(json.dumps(out, ensure_ascii=False) + "\n")

        generate_gold_file(test_df, "subtask_1_eng_laptop_golds.json")
    
if __name__ == "__main__":
    main()