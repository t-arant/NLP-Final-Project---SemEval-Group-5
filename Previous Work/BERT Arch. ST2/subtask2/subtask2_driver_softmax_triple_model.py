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

from subtask2_model_softmax_triple_model import (
    BERT_Tag_Model,
    BERT_Regression_Model, 
    VA_ASP_OPN_Split_Dataset, 
    eval_regressor_epoch,
    train_regressor_epoch,
    train_tagger_epoch, 
    eval_tagger_epoch, 
    model_inference_reg, 
    model_inference_tag,
    generate_results, 
    LABEL_TO_NUM
)
from data_loader_sub_2 import (
    generate_gold_file,
    load_local_json_to_df, 
    model_name
    # DATASET
)

from root_mse import rmse

from sklearn.utils.class_weight import compute_class_weight

DATASET = "/home/tony-arant/NLP Final Project/subtask2/subtask_2_eng_laptop_opn_asp_split_monolingual.json"

ASP_FLAG = 1
OPN_FLAG = 2
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

    # Creating ASP and OPN weights

    # ASP
    asp_labs = []

    asp_df = train_df["BIOES_asp"]

    for l in range(0, len(asp_df)):
        asp_labs.extend(asp_df.iloc[l])


    num_asp_lab = [LABEL_TO_NUM[label] for label in asp_labs]
    num_asp_lab = [x for x in num_asp_lab if x != LABEL_TO_NUM["EMB_P"]]
    # print(num_asp_lab)
    
    num_asp_lab_np = np.array(num_asp_lab)
    

    asp_weights = compute_class_weight(class_weight="balanced", classes=np.unique(num_asp_lab_np), y=num_asp_lab_np)
    asp_weights_torch = torch.from_numpy(asp_weights)

    

    # OPN

    opn_labs = []

    opn_df = train_df["BIOES_opn"]

    for l in range(0, len(opn_df)):
        opn_labs.extend(opn_df.iloc[l])


    num_opn_lab = [LABEL_TO_NUM[label] for label in opn_labs]
    num_opn_lab = [x for x in num_opn_lab if x != LABEL_TO_NUM["EMB_P"]]
    # print(num_asp_lab)
    
    num_opn_lab_np = np.array(num_opn_lab)
    opn_weights = compute_class_weight(class_weight="balanced", classes=np.unique(num_opn_lab_np), y=num_opn_lab_np)
    opn_weights_torch = torch.from_numpy(opn_weights)

    # runs if model is in training mode
    if train == 1:
        
        # Create tokenized train and dev data, and encapsulate into DataLoader 
        train_dataset = VA_ASP_OPN_Split_Dataset(train_df, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        dev_dataset = VA_ASP_OPN_Split_Dataset(dev_df, tokenizer)
        dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False)

        lr = 1e-5 #learning rate

        # sets desired epochs for training 
        # TODO: fix and automate
        epochs_tag = 8
        epochs_reg = 6
        
        # sets index of output loss graph data
        # TODO: fix and automate 
        trial = -100
        
        # Send model to device and define learning rate, epochs, optimizer function, and loss funct. 
        model_tag_asp = BERT_Tag_Model().to(device)
        lr = locals().get("lr", 1e-5)
        r_epochs = locals().get("epochs_reg", 5)
        t_epochs = locals().get("epochs_tag", 5)

        optimizer1 = torch.optim.AdamW(model_tag_asp.parameters(), lr=lr)
        va_loss_fn = rmse()

        asp_weights_torch = torch.from_numpy(asp_weights).float().to(device)
        opn_weights_torch = torch.from_numpy(opn_weights).float().to(device)
        tag_asp_loss_fn = nn.CrossEntropyLoss(ignore_index=LABEL_TO_NUM["EMB_P"], weight=asp_weights_torch)
        tag_opn_loss_fn = nn.CrossEntropyLoss(ignore_index=LABEL_TO_NUM["EMB_P"], weight=opn_weights_torch)

        # tag_opn_loss_fn = nn.CrossEntropyLoss(ignore_index=0, weight=asp_weights)

        # model freezing based on desired arch. 
        # TODO: fix this so that it can be automated with desired regression head arch. 
        
        # model.freeze_backbone()

        print("\n\nTraining Tagger ASP\n\n")
        train_losses_tag_asp = []

        # model.freeze_regressor()
        # model.freeze_backbone()
        for epoch in range(t_epochs):
            train_loss = train_tagger_epoch(model_tag_asp, train_loader, optimizer1, tag_asp_loss_fn, device, ASP_FLAG)
        
            # for validation loss, we can create a similar eval function
            val_loss = eval_tagger_epoch(model_tag_asp, dev_loader, tag_asp_loss_fn, device, ASP_FLAG)
        
            print(f"model:{model_name} Epoch:{epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")
            train_losses_tag_asp.append((epoch, train_loss, val_loss))


        torch.save(model_tag_asp.state_dict(), "triumverate_models/asp_tagger_state.pth")

        del model_tag_asp
        del optimizer1
        torch.cuda.empty_cache()

        model_tag_opn = BERT_Tag_Model().to(device)
        optimizer2 = torch.optim.AdamW(model_tag_opn.parameters(), lr=lr)
        

        print("\n\nTraining Tagger OPN\n\n")
        train_losses_tag_opn = []
        
        # model.freeze_regressor()
        # model.freeze_backbone()
        for epoch in range(t_epochs):
            train_loss = train_tagger_epoch(model_tag_opn, train_loader, optimizer2, tag_opn_loss_fn, device, OPN_FLAG)
        
            # for validation loss, we can create a similar eval function
            val_loss = eval_tagger_epoch(model_tag_opn, dev_loader, tag_opn_loss_fn, device, OPN_FLAG)
        
            print(f"model:{model_name} Epoch:{epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")
            train_losses_tag_opn.append((epoch, train_loss, val_loss))


        torch.save(model_tag_opn.state_dict(), "triumverate_models/opn_tagger_state.pth")
        del model_tag_opn
        del optimizer2
        torch.cuda.empty_cache()

        print("\n\nTraining Regressor\n\n")
        model_reg = BERT_Regression_Model().to(device)
        optimizer3 = torch.optim.AdamW(model_reg.parameters(), lr=lr)
        train_losses_reg = []
        
        # model.freeze_regressor()
        # model.freeze_backbone()
        for epoch in range(r_epochs):
            train_loss = train_regressor_epoch(model_reg, train_loader, optimizer3, va_loss_fn, device)
        
            # for validation loss, we can create a similar eval function
            val_loss = eval_regressor_epoch(model_reg, dev_loader, va_loss_fn, device)
        
            print(f"model:{model_name} Epoch:{epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")
            train_losses_reg.append((epoch, train_loss, val_loss))


        torch.save(model_reg.state_dict(), "triumverate_models/regressor_state.pth")
        torch.cuda.empty_cache()
        
        train_loader_whole = DataLoader(train_dataset, batch_size=100)
        dev_loader_whole = DataLoader(dev_dataset, batch_size=100)

        print("\n\nRegressor Model\nTrain/Dev Loss:")
        reg_train_loss_final  = model_inference_reg(model_reg,  va_loss_fn, train_loader_whole, device)

        reg_dev_loss_final  = model_inference_reg(model_reg, va_loss_fn, dev_loader_whole,  device)

        print(f"Reg: {reg_train_loss_final}\tTag: {reg_dev_loss_final}")


        del model_reg
        del optimizer3
        torch.cuda.empty_cache()


        tag_asp_model = BERT_Tag_Model().to(device)
        tag_asp_model.load_state_dict(torch.load("triumverate_models/asp_tagger_state.pth"))

        print("\n\nTagger Model ASP\nTrain/Dev Loss:")
        tag_asp_train_loss_final  = model_inference_tag(tag_asp_model, tag_asp_loss_fn, train_loader_whole,  device, ASP_FLAG)

        tag_asp_train_loss_final  = model_inference_tag(tag_asp_model, tag_asp_loss_fn, dev_loader_whole,  device, ASP_FLAG)

        print(f"Reg: {tag_asp_train_loss_final}\tTag: {tag_asp_train_loss_final}")


        del tag_asp_model
        torch.cuda.empty_cache()

        tag_opn_model = BERT_Tag_Model().to(device)
        tag_opn_model.load_state_dict(torch.load("triumverate_models/opn_tagger_state.pth"))

        print("\n\nTagger Model OPN\nTrain/Dev Loss:")
        tag_opn_train_loss_final  = model_inference_tag(tag_opn_model, tag_opn_loss_fn, train_loader_whole,  device, OPN_FLAG)

        tag_opn_train_loss_final  = model_inference_tag(tag_opn_model, tag_opn_loss_fn, dev_loader_whole,  device, OPN_FLAG)

        print(f"Reg: {tag_opn_train_loss_final}\tTag: {tag_opn_train_loss_final}")
        # final_train_loss_r, final_train_loss_t = model_inference(model, va_loss_fn, tag_loss_fn, train_loader_whole, device)
        # print(f"Reg: {final_train_loss_r}\tTag: {final_train_loss_t}")
        # print("Val Loss:")
        # final_val_loss_r, final_val_loss_t = model_inference(model, va_loss_fn, tag_loss_fn, dev_loader_whole, device)
        # print(f"Reg: {final_val_loss_r}\tTag: {final_val_loss_t}")

        # train_losses.append(("eval", final_train_loss, final_val_loss))

        # # place logged loss data into a DF, then write to CSV
        # # loss_df = pd.DataFrame(train_losses, columns=['Epoch', 'Train Loss', 'Validation Loss'])
        # # loss_df.to_csv(f"loss_graphs/subtask1_{trial}.csv", index=False)

        # # added this in to save state dictionary of final model, so we can load this and not train a new model for final eval
        # # TODO: only save model if training final model arch. 

        # torch.save(model.state_dict(), "english_laptop_final_subtask_2.pth")

    # Runs if model is in test mode
    # Returns a JSONL of VA predictions from text
    elif train == 0: 
        # load state dictionary and sent to GPU
        reg_path = "triumverate_models/regressor_state.pth"
        tag_asp_path = "triumverate_models/asp_tagger_state.pth"
        tag_opn_path = "triumverate_models/opn_tagger_state.pth"


        # Tokenize test datasets, and load into DataLoader
        # test_df.to_csv("datacontamination.csv", index=False)
        test_dataset = VA_ASP_OPN_Split_Dataset(test_df, tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        output = generate_results(reg_path, tag_asp_path, tag_opn_path, test_loader, device, test_df, tokenizer)
        
        # write predictions to orgainizer specified formate file
        with open("subtask_2_eng_laptop_preds_smax_tri.json", "w", encoding="utf-8") as f:
            for out in output:
               f.write(json.dumps(out, ensure_ascii=False) + "\n")

        generate_gold_file(test_df, "subtask_2_eng_laptop_golds_smax.json")
    
if __name__ == "__main__":
    main()