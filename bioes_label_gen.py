# This program takes in data for SemEval Task 3 subtask 1 tract A and produces a BIOES labeling scheme from given orthographic labels. BIOES tags are based on tokenizer. 

import requests
from transformers import AutoTokenizer
import json
from typing import List, Dict
import pandas as pd
import numpy as np
import torch
import ast

ASP_B = "ASP_B"
ASP_I = "ASP_I"
ASP_E = "ASP_E"
ASP_S = "ASP_S"
ASP_O = "ASP_O" 

# TRAIN_URL = f"https://raw.githubusercontent.com/DimABSA/DimABSA2026/refs/heads/main/task-dataset/track_a/{subtask}/{lang}/{lang}_{domain}_train_alltasks.jsonl"

def load_jsonl_url(url: str) -> List[Dict]:
    resp = requests.get(url)
    resp.raise_for_status()
    return [json.loads(line) for line in resp.text.splitlines()]

def jsonl_to_df(data):
    if 'Quadruplet' in data[0]:
        df = pd.json_normalize(data, 'Quadruplet', ['ID', 'Text'])
        df[['Valence', 'Arousal']] = df['VA'].str.split('#', expand=True).astype(float)
        df = df.drop(columns=['VA', 'Category', 'Opinion'])  # drop unnecessary columns
        df = df.drop_duplicates(subset=['ID', 'Aspect'], keep='first')  # remove duplicate ID+Aspect

    elif 'Triplet' in data[0]:
        df = pd.json_normalize(data, 'Triplet', ['ID', 'Text'])
        df[['Valence', 'Arousal']] = df['VA'].str.split('#', expand=True).astype(float)
        df = df.drop(columns=['VA', 'Opinion'])  # drop unnecessary columns
        df = df.drop_duplicates(subset=['ID', 'Aspect'], keep='first')  # remove duplicate ID+Aspect

    elif 'Aspect' in data[0]:
        df = pd.json_normalize(data, 'Aspect', ['ID', 'Text'])
        df = df.rename(columns={df.columns[0]: "Aspect"})  # rename to Aspect
        df['Valence'] = 0  # default value
        df['Arousal'] = 0  # default value

    else:
        raise ValueError("Invalid format: must include 'Quadruplet' or 'Triplet' or 'Aspect'")

    return df

def generate_BIOES_labesls(lang, task, subtask, domain, tokenizer, scheme):
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    train_url = f"https://raw.githubusercontent.com/DimABSA/DimABSA2026/refs/heads/main/task-dataset/track_a/{subtask}/{lang}/{lang}_{domain}_train_alltasks.jsonl"
    # predict_url = f"https://raw.githubusercontent.com/DimABSA/DimABSA2026/refs/heads/main/task-dataset/track_a/{subtask}/{lang}/{lang}_{domain}_dev_{task}.jsonl"
    
    train_raw = load_jsonl_url(train_url)
    train_df = jsonl_to_df(train_raw)

    train_df = train_df.reset_index(drop=True)
    train_df["BIOES"] = None
    
    with pd.option_context('display.max_columns', None):
        print(train_df.head())

    # print(f"!!!!!!!!!!!!!!!!!!!!!!\n{len(train_df)}")
    # idxes = train_df.index.tolist()
    for x in range(len(train_df)):
        labels = []

        text = train_df.iloc[x]["Text"]
        text_enc = tokenizer(text, truncation=True, padding="max_length", max_length=128,return_tensors="pt")
        sentence_enc = text_enc["input_ids"][0]
        # cls_i = torch.where(sentence_enc == 101)
        # sep_i = torch.where(sentence_enc == 102)

        aspect_term = train_df.iloc[x]["Aspect"]
        asp_enc = tokenizer(aspect_term, truncation = True, max_length = 128, return_tensors="pt")
        asp_enc = asp_enc["input_ids"][0]
        asp_enc = asp_enc[1:-1] # removing CLS and SEP tokens

        mask = torch.zeros_like(sentence_enc, dtype=torch.bool)
        asp_len = len(asp_enc)
        for i in range(len(sentence_enc) - asp_len + 1):
            if torch.all(sentence_enc[i:i+asp_len] == asp_enc):
                mask[i:i+asp_len] = True
        # indices = torch.where(mask)
        
        indices = torch.where(mask)[0]

        if len(indices) >= 2:
            seen_tokens = set()
            good_indexes = []

            for indx in range(len(indices)):
                token_id = sentence_enc[indices[indx]].item()  
                if token_id not in seen_tokens:
                    seen_tokens.add(token_id)
                    good_indexes.append(indx)  


            
            good_indexes = torch.tensor(good_indexes, dtype=torch.long)
            indices = indices[good_indexes]
                

        # tokens = []

        # if len(indices) >= 2:
        #     for idx in range(0, len(indices)):
        #         tokens.append(sentence_enc[idx])

        #     stride = 0
        #     j = 1

        #     while(j < len(indices)):
        #         if sentence_enc[indices[0]] != sentence_enc[indices[j]]:
        #             stride += 1 
        #             j += 1
        #         else: 
        #             break
            
        #     z = stride
        #     deletes = []

        #     pointer = 0
        #     counts = 0

        #     while z < len(indices):
        #         if counts == (stride - 1):
        #             pointer = 0
        #             counts = 0

        #         if sentence_enc[indices[pointer]] == sentence_enc[indices[z]]:
        #             deletes.append(z)
        #             pointer += 1
        #             counts += 1
                
        #     indices = [v for i, v in enumerate(indices) if i not in deletes]

        # print(indices)
        if len(indices) == 0:
            labels = [ASP_O] * len(mask)
        
        else:
            for ind in range(len(sentence_enc)):
                if ind not in indices:
                    labels.append(ASP_O)
                elif ind in indices:
                    if ind == 0:
                        if (ind + 1) in indices:
                            labels.append(ASP_B)
                        else:
                            labels.append(ASP_S)
                    elif ind == len(sentence_enc) - 1:
                        if (ind - 1) in indices:
                            labels.append(ASP_E)
                        else:
                            labels.append(ASP_S)
                    elif ((ind - 1) in indices) and ((ind + 1) in indices):
                        labels.append(ASP_I)
                    elif ((ind - 1) in indices):
                        labels.append(ASP_E)
                    elif ((ind + 1) in indices):
                        labels.append(ASP_B)
                    else:
                        labels.append(ASP_S)

                        
            
        # for i, row in train_df.iterrows():
        #     assert len(row["BIOES"]) == 128, f"Row {i} has wrong label length!"
                    

        # else:
        #     prev_flag = "F"
        #     for m in range(len(mask)):
        #         if mask[m] == False:
        #             labels.append(ASP_O)
        #             prev_flag = "F"
        #         else:
        #             if prev_flag == "F":
        #                 if m == len(mask)-1:
        #                     labels.append(ASP_S)
        #                     prev_flag = "F"

        #                 elif mask[m+1] == True:
        #                     labels.append(ASP_B)
        #                     prev_flag = "B"

        #                 elif mask[m+1] == False:
        #                     labels.append(ASP_S)
        #                     prev_flag = "F"

        #             elif prev_flag == "B":
        #                 if m == len(mask) - 1:
        #                     labels.append(ASP_E)
        #                     prev_flag = "F"
                        
        #                 elif mask[m+1] == False:
        #                     labels.append(ASP_E)
        #                     prev_flag = "F"

        #                 elif mask[m+1] == True:
        #                     labels.append(ASP_I)
        #                     prev_flag = "I"
                    
        #             elif prev_flag == "I":
        #                 if m == len(mask) - 1:
        #                     labels.append(ASP_E)
        #                     prev_flag = "F"
                        
        #                 elif mask[m+1] == True:
        #                     labels.append(ASP_I)
        #                     prev_flag = "I"

        #                 elif mask[m+1] == False:
        #                     labels.append(ASP_E)
        #                     prev_flag = "F"


                
        train_df.at[x, "BIOES"] = labels
        # train_df['BIOES'] = train_df['BIOES'].apply(ast.literal_eval)
    train_df.to_json("test_lab.json", orient="records", lines=True)


# At the moment this test only works for a single aspect term, as ASP_B , ASP_I, ASP_S, and ASP_E sequence components are not taken into account
def test_labeling(df, tknzr):
    correct_count = 0
    incorrect_count = 0

    incorrect_indexes = []
    tokenizer = AutoTokenizer.from_pretrained(tknzr)

    with open("incorrect_reconstructions.txt", "w") as f:

        for x in range(len(df)):
            text = df.iloc[x]["Text"]
            enc_lab = df.iloc[x]["BIOES"]
            aspect = df.iloc[x]['Aspect']

            enc_text = tokenizer(text, truncation=True, padding="max_length", max_length=128,return_tensors="pt")
            enc_text = enc_text["input_ids"][0]

            text_tokens = []

            for i in range(len(enc_text)):

                if (enc_lab[i] == ASP_B) or (enc_lab[i] == ASP_E) or (enc_lab[i] == ASP_I) or (enc_lab[i] == ASP_S):
                    text_tokens.append(enc_text[i])
            
            text_tensor = torch.tensor(text_tokens)

            reconstructed_text = tokenizer.decode(text_tensor, skip_special_tokens=False, clean_up_tokenization_spaces=False)


            if reconstructed_text == "":
                reconstructed_text = "NULL"

            if reconstructed_text == aspect:
                correct_count += 1
            else: 
                incorrect_count += 1 
                incorrect_indexes.append(x)
                
                f.write(f"{x}:\t")
                f.write(f"{reconstructed_text}\t|\t")
                f.write(f"{aspect}\n")

                # print(f"\"{reconstructed_text}\"\t\"{aspect}\"")

            
    print(f"corrects: {correct_count}\tincorrects {incorrect_count}")

    # print(incorrect_indexes)

        

        



subtask = "subtask_1"#don't change
task = "task1"#don't change
lang = "eng" #chang the language you want to test
domain = "laptop" #change what domain you want to test
tokenizer = "distilbert-base-multilingual-cased"

generate_BIOES_labesls(lang, task, subtask, domain, tokenizer, "bioes")

aug_df_bioes_lab = df = pd.read_json("test_lab.json", orient="records", lines=True)

test_labeling(aug_df_bioes_lab, tokenizer)

