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

from bioes_classification_asp_opn_crf_sep import (
    TransformerTagger, 
    TransformerRegressor, 
    VA_ASP_OPN_Split_Dataset, 
    # eval_regressor_epoch,
    # train_regressor_epoch,
    # train_tagger_epoch, 
    # eval_tagger_epoch, 
    # model_inference, 
    data_t_t_split,
    load_local_json_to_df,
    generate_preds_dev,
    eval_regressor_epoch,
    generate_final_predictions,
    # MODEL_NAME, 
    # DATASET
)

from data_loader_sub_2 import (
    generate_gold_file
)
ASP_FLAG = 1 
OPN_FLAG = 0
# from data_loader_sub_2 import (
#     generate_gold_file,
#     load_local_json_to_df, 
#     model_name,
#     DATASET
# )

from root_mse import rmse


DATASET = "/home/tony-arant/NLP Final Project/data_labs/separate_asp_opn/distilbert/distilbert_subtask_2_ukr_restaurant_sep.json" 
MODEL_NAME = "distilbert-base-multilingual-cased"

MODEL_STATE_FILE_ASP = "ukr_rest_distilbert_comb_state_asp.pth"
MODEL_STATE_FILE_OPN = "ukr_rest_distilbert_comb_state_opn.pth"
MODEL_STATE_FILE_REG = "ukr_rest_distilbert_comb_state_reg.pth"

PREDS_FILE_NAME = "subtask_2_ukr_rest_preds_crf_sep_distilbert.json"
GOLDS_FILE_NAME = "subtask_2_ukr_rest_golds_crf_sep_distilbert.json"

LANG = "ukr"

# Takes in DataFrame, splits based on ID label of train, dev, or test
# returns split data in 3 DataFrames
# def data_t_t_split(dataset):
#     train_mask = dataset['ID'].str.contains("train", case=False, na=False)
#     train_df = dataset[train_mask].copy().reset_index(drop=True)

#     dev_mask = dataset['ID'].str.contains("dev", case=False, na=False)
#     dev_df = dataset[dev_mask].copy().reset_index(drop=True)

#     test_mask = dataset['ID'].str.contains("test", case=False, na=False)
#     test_df = dataset[test_mask].copy().reset_index(drop=True)

#     return train_df, dev_df, test_df


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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Define Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # runs if model is in training mode
    if train == 1:
        
        # Create tokenized train and dev data, and encapsulate into DataLoader 
        train_dataset = VA_ASP_OPN_Split_Dataset(train_df, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        dev_dataset = VA_ASP_OPN_Split_Dataset(dev_df, tokenizer)
        dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False)

        lr_tag = 1e-4 #learning rate
        lr_reg = 1e-5 #learning rate


        # sets desired epochs for training 
        # TODO: fix and automate
        epochs_tag_asp = 5
        epochs_tag_opn = 6
        epochs_reg = 8
        
        # sets index of output loss graph data
        # TODO: fix and automate 
        trial = -100
        
        # Send model to device and define learning rate, epochs, optimizer function, and loss funct. 
        # model_asp = TransformerASPTagger().to(device)
        lr_t = locals().get("lr_tag", 1e-5)
        lr_r = locals().get("lr_reg", 1e-5)

        r_epochs = locals().get("epochs_reg", 5)
        ta_epochs = locals().get("epochs_tag_asp", 5)
        to_epochs = locals().get("epochs_tag_opn", 5)
        
        
        # optimizer_opn = torch.optim.AdamW(model.parameters(), lr=lr_t)
        va_loss_fn = rmse()

        # model freezing based on desired arch. 
        # TODO: fix this so that it can be automated with desired regression head arch. 
        
        # model.freeze_backbone()

        print("\n\nTraining ASP Tagger\n\n")
        train_losses_tag_asp = []

        model_asp = TransformerTagger(MODEL_NAME, 0.1).to(device)
        optimizer_asp = torch.optim.AdamW(model_asp.parameters(), lr=lr_t)
        
        # model.freeze_backbone()
        for epoch in range(epochs_tag_asp):
            train_loss = model_asp.train_epoch_tagger(train_loader, optimizer_asp, device, ASP_FLAG)
        
            # for validation loss, we can create a similar eval function
            val_loss = model_asp.eval_epoch(dev_loader, device, ASP_FLAG)
        
            print(f"model:{MODEL_NAME} Epoch:{epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")
            train_losses_tag_asp.append((epoch, train_loss, val_loss))

        print("\nTrain Loss:")
        asp_train_loss = model_asp.eval_epoch(train_loader, device, ASP_FLAG)
        print(f"{asp_train_loss}")
        print("Val Loss:")
        asp_dev_loss = model_asp.eval_epoch(dev_loader, device, ASP_FLAG)
        print(f"{asp_dev_loss}")
        # preds = generate_preds_dev(model_asp, dev_loader, device, dev_df, tokenizer, ASP_FLAG)

    # print(preds)

        # with open("extracted_asp_dev.txt", "w") as f:
        #     f.write(f"extracted\tactual\ttext\n")
        #     for p in preds:
        #         f.write(f"{p[0]} | \t")
        #         f.write(f"{p[1]} | \t")
        #         f.write(f"{p[2]}\n")

        torch.save(model_asp.state_dict(), MODEL_STATE_FILE_ASP)
        del model_asp
        del optimizer_asp
        torch.cuda.empty_cache()


        print("\n\nTraining OPN Tagger\n\n")
        train_losses_tag_asp = []

        model_opn = TransformerTagger(MODEL_NAME, 0.1).to(device)
        optimizer_opn = torch.optim.AdamW(model_opn.parameters(), lr=lr_t)

        # model.freeze_backbone()
        for epoch in range(epochs_tag_opn):
            train_loss = model_opn.train_epoch_tagger(train_loader, optimizer_opn, device, OPN_FLAG)

            # for validation loss, we can create a similar eval function
            val_loss = model_opn.eval_epoch(dev_loader, device, OPN_FLAG)

            print(f"model:{MODEL_NAME} Epoch:{epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")
            train_losses_tag_asp.append((epoch, train_loss, val_loss))

        print("\nTrain Loss:")
        asp_train_loss = model_opn.eval_epoch(train_loader, device, OPN_FLAG)
        print(f"{asp_train_loss}")
        print("Val Loss:")
        asp_dev_loss = model_opn.eval_epoch(dev_loader, device, OPN_FLAG)
        print(f"{asp_dev_loss}")
        # preds = generate_preds_dev(model_opn, dev_loader, device, dev_df, tokenizer, OPN_FLAG)

        # print(preds)

        # with open("extracted_opn_dev.txt", "w") as f:
        #     f.write(f"extracted\tactual\ttext\n")
        #     for p in preds:
        #         f.write(f"{p[0]} | \t")
        #         f.write(f"{p[1]} | \t")
        #         f.write(f"{p[2]}\n")

        torch.save(model_opn.state_dict(), MODEL_STATE_FILE_OPN)
        del model_opn
        del optimizer_opn
        torch.cuda.empty_cache()


        print("\n\nTraining Regressor\n\n")
        train_losses_reg = []

        model_reg = TransformerRegressor(MODEL_NAME, 0.1).to(device)
        optimizer_reg = torch.optim.AdamW(model_reg.parameters(), lr=lr_t)

        # model_reg.freeze_backbone()
        for epoch in range(r_epochs):
            train_loss = model_reg.train_regressor_epoch(train_loader, optimizer_reg, va_loss_fn, device)

            # for validation loss, we can create a similar eval function
            val_loss = eval_regressor_epoch(model_reg, dev_loader, va_loss_fn, device)

            print(f"model:{MODEL_NAME} Epoch:{epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")
            train_losses_reg.append((epoch, train_loss, val_loss))

        print("\nTrain Loss:")
        reg_train_loss = eval_regressor_epoch(model_reg, train_loader, va_loss_fn, device)
        print(f"{reg_train_loss}")
        print("Val Loss:")
        reg_dev_loss = eval_regressor_epoch(model_reg, dev_loader, va_loss_fn, device)
        print(f"{reg_dev_loss}")


        torch.save(model_reg.state_dict(), MODEL_STATE_FILE_REG)
        del model_reg
        del optimizer_reg
        torch.cuda.empty_cache()


    # Runs if model is in test mode
    # Returns a JSONL of VA predictions from text
    elif train == 0: 
        # reg_path = "triumverate_models/regressor_state.pth"
        # tag_asp_path = "triumverate_models/asp_tagger_state.pth"
        # tag_opn_path = "triumverate_models/opn_tagger_state.pth"


        # Tokenize test datasets, and load into DataLoader
        # test_df.to_csv("datacontamination.csv", index=False)
        test_dataset = VA_ASP_OPN_Split_Dataset(test_df, tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        output = generate_final_predictions(MODEL_STATE_FILE_REG, MODEL_STATE_FILE_ASP, MODEL_STATE_FILE_OPN, MODEL_NAME, test_loader, device, test_df, tokenizer, LANG)
        
        # write predictions to orgainizer specified formate file
        with open(PREDS_FILE_NAME, "w", encoding="utf-8") as f:
            for out in output:
               f.write(json.dumps(out, ensure_ascii=False) + "\n")

        generate_gold_file(test_df, GOLDS_FILE_NAME)
    
if __name__ == "__main__":
    main()