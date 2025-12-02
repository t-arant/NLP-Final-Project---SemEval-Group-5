
# This file contains helper functions assocated with loading data for the project 

import json
import pandas as pd

DATASET = "//home/tony-arant/NLP Final Project/subtask2_data/subtask_2_eng_laptop.json"

model_name = "bert-base-cased" # chage your transformer model

# This function will load a local JSON file and translate it to a dataframe
# Input: file path 
# Output DataFrame
def load_local_json_to_df(file_name):
     df = pd.read_json(file_name, orient="records", lines=True)
     return df 

# This function takes in the Gold Labels dataframe, and produces a JSONL of Gold Label Data. Data is saved in file. 
# Input: DataFrame of Gold Data, file path 
def generate_gold_file(data, file_path):
     data.to_csv("dc2.csv")
     d_c = data.copy()
     combined_labels = []

     # combine Valence and arousal Scores into "VA#AR" str
     for i in range(len(d_c)):
        score1 = f'{d_c.iloc[i]["Valence"]:.2f}'
        score2 = f'{d_c.iloc[i]["Arousal"]:.2f}'
        combined_labels.append(f"{score1}#{score2}")

     d_c['VA'] = combined_labels

     # Group data by ID, and nest multiple Apect-VA_Score pairs in "Aspect_VA" structure   
     golds = [
    {
        "ID": group_name,
        # "Text": group["Text"].iloc[0],  # uncomment if you want to include text
        "Triplet": group[["Aspect", "Opinion", "VA"]].to_dict(orient="records")
    }
    for group_name, group in d_c.groupby("ID")
     ]
     
     # Write to 
     with open(file_path, "w", encoding="utf-8") as f:
          for entry in golds:
               f.write(json.dumps(entry, ensure_ascii=False) + "\n")