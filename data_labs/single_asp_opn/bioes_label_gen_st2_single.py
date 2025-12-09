# This program takes in data for SemEval Task 3 subtask 1 tract A and produces a BIOES labeling scheme from given orthographic labels. BIOES tags are based on tokenizer. 

# Nov 28 - modified labels to match len of attention mask

import requests
from transformers import AutoTokenizer
import json
from typing import List, Dict
import pandas as pd
import numpy as np
import torch
import ast
import re

# Definition of Gloabal Labels
# ASP_I = "ASP_I"
# OPN_I = "OPN_I"
# EMB_O = "EMB_O" 
# EMB_P = "EMB_P"

ASP_B = "ASP_B"
ASP_I = "ASP_I"
ASP_E = "ASP_E"
ASP_S = "ASP_S"

OPN_B = "OPN_B"
OPN_I = "OPN_I"
OPN_E = "OPN_E"
OPN_S = "OPN_S"

EMB_O = "EMB_O" 
EMB_P = "EMB_P"

# loads Task Data from organizer Github - Taken from starter Kit
def load_jsonl_url(url: str) -> List[Dict]:
    resp = requests.get(url)
    resp.raise_for_status()
    return [json.loads(line) for line in resp.text.splitlines()]

# Transforms data from JSON to Python-friendly DataFrame - Taken from starter Kit
def jsonl_to_df(data):
    if 'Quadruplet' in data[0]:
        df = pd.json_normalize(data, 'Quadruplet', ['ID', 'Text'])
        df[['Valence', 'Arousal']] = df['VA'].str.split('#', expand=True).astype(float)
        df = df.drop(columns=['VA', 'Category',])  # drop unnecessary columns
        df = df.drop_duplicates(subset=['ID', 'Aspect'], keep='first')  # remove duplicate ID+Aspect

    elif 'Triplet' in data[0]:
        df = pd.json_normalize(data, 'Triplet', ['ID', 'Text'])
        df[['Valence', 'Arousal']] = df['VA'].str.split('#', expand=True).astype(float)
        df = df.drop(columns=['VA'])  # drop unnecessary columns
        df = df.drop_duplicates(subset=['ID', 'Aspect'], keep='first')  # remove duplicate ID+Aspect

    elif 'Aspect' in data[0]:
        df = pd.json_normalize(data, 'Aspect', ['ID', 'Text'])
        df = df.rename(columns={df.columns[0]: "Aspect"})  # rename to Aspect
        df['Valence'] = 0  # default value
        df['Arousal'] = 0  # default value

    else:
        raise ValueError("Invalid format: must include 'Quadruplet' or 'Triplet' or 'Aspect'")

    return df

# This function takes in the string label of aspect, and the training data and creates a gold BIOES label for the text. 
# The BIOES label is based on the tokenized aspect and training text

def generate_BIOES_labesls(lang, task, subtask, domain, tkrzer_name, scheme, file_name, max_len):
    
    tokenizer = AutoTokenizer.from_pretrained(tkrzer_name)
    
    # Get Raw Data
    if lang == 'jpn':
        train_url = "https://raw.githubusercontent.com/DimABSA/DimABSA2026/refs/heads/main/task-dataset/track_a/subtask_1/jpn/jpn_hotel_train_alltasks.jsonl"
    
    else: 
        train_url = f"https://raw.githubusercontent.com/DimABSA/DimABSA2026/refs/heads/main/task-dataset/track_a/{subtask}/{lang}/{lang}_{domain}_train_alltasks.jsonl"

    
    # Transform data to DF
    train_raw = load_jsonl_url(train_url)
    train_df = jsonl_to_df(train_raw)

    # Adding I/O label column
    train_df = train_df.reset_index(drop=True)
    train_df["BIOES"] = None
    
    with pd.option_context('display.max_columns', None):
        print("PRINTING HEAD---------------------------------------------------------------------")
        print(train_df.head())

    for x in range(len(train_df)):
        labels = []

        # Tokenize Data
        text = train_df.iloc[x]["Text"]
        text_enc = tokenizer(text, truncation=True, padding="max_length", max_length=max_len,return_tensors="pt")
        sentence_enc = text_enc["input_ids"][0]


        aspect_term = train_df.iloc[x]["Aspect"]
        asp_tokens = tokenizer(aspect_term, truncation = True, max_length = max_len, return_tensors="pt")
        asp_enc = asp_tokens["input_ids"][0]
        asp_enc = asp_enc[1:-1] # removing CLS and SEP tokens

        opinion_term = train_df.iloc[x]["Opinion"]
        opinion_tokens = tokenizer(opinion_term, truncation = True, max_length = max_len, return_tensors="pt")
        opn_enc = opinion_tokens["input_ids"][0]
        opn_enc = opn_enc[1:-1] # removing CLS and SEP tokens

        sent_ind = 0
        asp_ind = 0
        opn_ind = 0

        asp_indices = []
        opn_indices = []
        
        asp_len = len(asp_enc)
        opn_len = len(opn_enc)

        while sent_ind < len(sentence_enc):
            if asp_enc[asp_ind] == sentence_enc[sent_ind]:
                asp_indices.append(sent_ind)
                asp_ind += 1

                if asp_ind >= asp_len:
                    asp_ind = 0

            if opn_enc[opn_ind] == sentence_enc[sent_ind]:
                opn_indices.append(sent_ind)
                opn_ind += 1

                if opn_ind >= opn_len:
                    opn_ind = 0

            sent_ind += 1
        

        # This is a check for duplicates. The apesct term can appear multiple times in the training sentence. For the purposes of task 1, we only need the 
        # single aspect and opinion term
        if len(asp_indices) >= asp_len:
            non_dupulicate_indices_asp = []
            for i_pointer in range(0, asp_len):
                non_dupulicate_indices_asp.append(i_pointer)
            
            asp_indices = [asp_indices[i] for i in non_dupulicate_indices_asp]

        if len(opn_indices) >= opn_len:
            non_dupulicate_indices_opn = []
            for i_pointer in range(0, opn_len):
                non_dupulicate_indices_opn.append(i_pointer)
            
            opn_indices = [opn_indices[i] for i in non_dupulicate_indices_opn]
 
        # I/O Labeling 
    #     if (len(asp_indices) == 0) and (len(opn_indices) == 0):
    #         for ind in range(len(sentence_enc)):
    #             if sentence_enc[ind] != 0:
    #                 labels.append(EMB_O)
    #             else: 
    #                 labels.append(EMB_P)

    #     else:
    #         for ind in range(len(sentence_enc)):
    #             if (ind not in asp_indices) and (ind not in opn_indices):
    #                 if sentence_enc[ind] != 0 and (sentence_enc[ind] != 101) and (sentence_enc[ind] != 102):
    #                     labels.append(EMB_O)
    #                 else: 
    #                     labels.append(EMB_P)

    #             if ind in asp_indices:
    #                 labels.append(ASP_I)
    #             elif ind in opn_indices: 
    #                 labels.append(OPN_I)

                
    #     train_df.at[x, "IO"] = labels
    # train_df.to_json(file_name, orient="records", lines=True)

    #     # If language is Chinese, indexes of where the aspect is generated by looping through text tokens and checking against aspect tokens
    #     if lang == 'zho':
    #         asp_len = len(asp_enc)

    #         indices = []
    #         index = 0
    #         asp_index = 0
    #         while index < len(sentence_enc):

    #             if asp_enc[asp_index] == sentence_enc[index]:
    #                 indices.append(index)
    #                 asp_index += 1 

    #                 if asp_index >= asp_len:
    #                     asp_index = 0

    #             index += 1 

    #     # If language is not Chinese, indexes of where the aspect is generated by creating a mask, then looping through data, checking against the mask. 
        # else:    
        #     mask = torch.zeros_like(sentence_enc, dtype=torch.bool)
        #     asp_len = len(asp_enc)
        #     for i in range(len(sentence_enc) - asp_len + 1):
        #         if torch.all(sentence_enc[i:i+asp_len] == asp_enc):
        #             mask[i:i+asp_len] = True
        #     indices = torch.where(mask)[0]
        
        # # This is a check for duplicates. The apesct term can appear multiple times in the training sentence. For the purposes of task 1, we only need the 
        # # single aspect term
        # if len(indices) >= asp_len:
        #     non_dupulicate_indices = []
        #     for i_pointer in range(0, asp_len):
        #         non_dupulicate_indices.append(i_pointer)
            
        #     indices = [indices[i] for i in non_dupulicate_indices]

        # If the aspect term is null, label all text tokens as outside, and padding tokens Pad
        if (len(asp_indices) == 0) and (len(opn_indices) == 0):
            for ind in range(len(sentence_enc)):
                if sentence_enc[ind] != 0:
                    labels.append(EMB_O)
                else: 
                    labels.append(EMB_P)

        # The labels are generated by checking the index list for postions in the tokenized text that match the aspect. 
        # Lookahead and lookbehind checks are done to determine the Beggining, inside, and end aspect labels.
        else:
            for ind in range(len(sentence_enc)):
                if (ind not in asp_indices) and (ind not in opn_indices):
                    # if sentence_enc[ind] != 0 and (sentence_enc[ind] != 101) and (sentence_enc[ind] != 101):
                    if sentence_enc[ind] != 0:

                        labels.append(EMB_O)
                    else: 
                        labels.append(EMB_P)
                elif ind in asp_indices:
                    if ind == 0:
                        if (ind + 1) in asp_indices:
                            labels.append(ASP_B)
                        else:
                            labels.append(ASP_S)
                    elif ind == len(sentence_enc) - 1:
                        if (ind - 1) in asp_indices:
                            labels.append(ASP_E)
                        else:
                            labels.append(ASP_S)
                    elif ((ind - 1) in asp_indices) and ((ind + 1) in asp_indices):
                        labels.append(ASP_I)
                    elif ((ind - 1) in asp_indices):
                        labels.append(ASP_E)
                    elif ((ind + 1) in asp_indices):
                        labels.append(ASP_B)
                    else:
                        labels.append(ASP_S)

                elif ind in opn_indices:
                    if ind == 0:
                        if (ind + 1) in opn_indices:
                            labels.append(OPN_B)
                        else:
                            labels.append(OPN_S)
                    elif ind == len(sentence_enc) - 1:
                        if (ind - 1) in opn_indices:
                            labels.append(OPN_E)
                        else:
                            labels.append(OPN_S)
                    elif ((ind - 1) in opn_indices) and ((ind + 1) in opn_indices):
                        labels.append(OPN_I)
                    elif ((ind - 1) in opn_indices):
                        labels.append(OPN_E)
                    elif ((ind + 1) in opn_indices):
                        labels.append(OPN_B)
                    else:
                        labels.append(OPN_S)

        
        train_df.at[x, "BIOES"] = labels
    train_df.to_json(file_name, orient="records", lines=True)

# In order to test if the generated BIOES labels are correct, we run the following test:
# Text is tokenized, and a for loop runs through the embeddings and checks each token label. 
# If the current label is a valid aspect label, that embedding is saved. The final embeddings are then reconstructed, and comapred against the string aspect label. 
#  
# At the moment this test only works for a single aspect term, as ASP_B , ASP_I, ASP_S, and ASP_E sequence components are not taken into account
def test_labeling(df, tknzr, lang, domain, max_tok, tok_type):
    # defnition of tokenizer and count variables
    correct_count = 0
    incorrect_count = 0

    incorrect_indexes = []
    tokenizer = AutoTokenizer.from_pretrained(tknzr)

    # I write the dataframe index, dataset index, incorrect reconstruction and string aspect label to a file for debuging. 
    with open(f"incorrect_reconstructions_{lang}_{domain}_{tok_type}_comb.txt", "w") as f:
        f.write(f"DF Index\t Dataset Index\tRecon Asp\tRecon Opn\t|\t ASP\tOPN\n")
        for x in range(len(df)):
            text = df.loc[x, "Text"]
            # enc_lab = df.loc[x, "BIOES"]
            enc_lab = df.loc[x, "BIOES"]

            aspect = df.loc[x, 'Aspect']
            opinion = df.loc[x, "Opinion"]

            enc_text = tokenizer(text, truncation=True, padding="max_length", max_length=max_tok,return_tensors="pt")
            enc_text = enc_text["input_ids"][0]

            asp_tokens = []
            opn_tokens = []


            for i in range(len(enc_lab)):

                # if (enc_lab[i] == ASP_I):
                if (enc_lab[i] == ASP_B) or (enc_lab[i] == ASP_I) or (enc_lab[i] == ASP_E) or (enc_lab[i] == ASP_S):
                    asp_tokens.append(enc_text[i])

                # if (enc_lab[i] == OPN_I):
                if (enc_lab[i] == OPN_B) or (enc_lab[i] == OPN_I) or (enc_lab[i] == OPN_E) or (enc_lab[i] == OPN_S):
                    opn_tokens.append(enc_text[i])
            
            opn_tensor = torch.tensor(opn_tokens)
            asp_tensor = torch.tensor(asp_tokens)


            reconstructed_opinion = tokenizer.decode(opn_tensor, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            reconstructed_aspect= tokenizer.decode(asp_tensor, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            
            # BERT tokenization will sometimes add extranious spaces to reconstructions of languages other than English
            # The string reconstruction needs to be post-processed in order to match the string aspect Label 
            if lang == "rus":
                reconstructed_opinion = re.sub(r' - ', '-', reconstructed_opinion)
                reconstructed_opinion  = re.sub(r'([0-9]) \+', r'\1+', reconstructed_opinion)
                reconstructed_opinion  = re.sub(r'([0-9]) %',  r'\1%', reconstructed_opinion)
                reconstructed_opinion = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', reconstructed_opinion)

                reconstructed_aspect = re.sub(r' - ', '-', reconstructed_aspect)
                reconstructed_aspect   = re.sub(r'([0-9]) \+', r'\1+', reconstructed_aspect)
                reconstructed_aspect   = re.sub(r'([0-9]) %',  r'\1%', reconstructed_aspect)
                reconstructed_aspect = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', reconstructed_aspect)
            
            if lang == "tat":
                reconstructed_opinion = re.sub(r' - ', '-', reconstructed_opinion)
                reconstructed_opinion = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', reconstructed_opinion)
                reconstructed_opinion  = re.sub(r'([0-9]) \+', r'\1+', reconstructed_opinion)
                reconstructed_opinion  = re.sub(r'([0-9]) %',  r'\1%', reconstructed_opinion)

                reconstructed_aspect = re.sub(r' - ', '-', reconstructed_aspect)
                reconstructed_aspect = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', reconstructed_aspect)
                reconstructed_aspect   = re.sub(r'([0-9]) \+', r'\1+', reconstructed_aspect)
                reconstructed_aspect   = re.sub(r'([0-9]) %',  r'\1%', reconstructed_aspect)
            
            if lang == "ukr":
                reconstructed_opinion = re.sub(r' - ', '-', reconstructed_opinion)
                reconstructed_opinion = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', reconstructed_opinion)
                reconstructed_opinion = re.sub(r' \' ', '\'', reconstructed_opinion)
                reconstructed_opinion  = re.sub(r'([0-9]) \+', r'\1+', reconstructed_opinion)
                reconstructed_opinion  = re.sub(r'([0-9]) %',  r'\1%', reconstructed_opinion)

                reconstructed_aspect = re.sub(r' - ', '-', reconstructed_aspect)
                reconstructed_aspect = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', reconstructed_aspect)
                reconstructed_aspect = re.sub(r' \' ', '\'', reconstructed_aspect)
                reconstructed_aspect   = re.sub(r'([0-9]) \+', r'\1+', reconstructed_aspect)
                reconstructed_aspect   = re.sub(r'([0-9]) %',  r'\1%', reconstructed_aspect)

            if lang == "zho":
                reconstructed_opinion = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', reconstructed_opinion)
                reconstructed_opinion = re.sub(r"\s+", "", reconstructed_opinion)

                reconstructed_aspect = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', reconstructed_aspect)
                reconstructed_aspect = re.sub(r"\s+", "", reconstructed_aspect)

            # If sentence contains no aspect term, the comparison must be with a string NULL
            if reconstructed_aspect == "":
                reconstructed_aspect = "NULL"

            if reconstructed_opinion == "":
                reconstructed_opinion = "NULL"

            if (reconstructed_aspect == aspect) and (reconstructed_opinion == opinion):
                correct_count += 1
            else: 
                incorrect_count += 1 

                if "test" not in str(df.iloc[x]['ID']):
                    incorrect_indexes.append(x)
                
                f.write(f"{x} - {df.iloc[x]['ID']}:\t")
                f.write(f"{reconstructed_aspect}\t{reconstructed_opinion}\t|\t")
                f.write(f"{aspect}\t{opinion}\n")

        f.write(f"\n\ncorrects: {correct_count}\tincorrects {incorrect_count}")
        
    df.to_csv(f"{lang}_{domain}_{tok_type}_comb.csv")

    with open(f"{lang}_{domain}_{tok_type}_drops_comb.txt", "w") as f2:
        f2.write("[")

        for idx in incorrect_indexes:
            f2.write(f"{idx},")
        f2.write("]")

    print(f"corrects: {correct_count}\tincorrects {incorrect_count}")
    

        

        


def main(): 
    subtask = "subtask_2"#don't change
    task = "task1"#don't change
    lang = "eng" #chang the language you want to test
    domain = "laptop" #change what domain you want to test
    tokenizer = "distilbert-base-multilingual-cased"
    f_name = "subtask_2_eng_laptop_nopad.json"

    generate_BIOES_labesls(lang, task, subtask, domain, tokenizer, "bioes", f_name, 128)

    aug_df_io_lab = pd.read_json(f_name, orient="records", lines=True)

    test_labeling(aug_df_io_lab, tokenizer, lang, domain, 128, "distilbert")


if __name__ == "__main__":
    main()

