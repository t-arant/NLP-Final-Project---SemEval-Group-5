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

DATASET = "/home/tony-arant/NLP Final Project/subtask_a_eng_laptop.json"

model_name = "distilbert-base-multilingual-cased" # chage your transformer model

def load_local_json_to_df(file_name):
     df = pd.read_json(file_name, orient="records", lines=True)
     return df 