## Subtask 1
Code and Data for Subtask 1, using BERT-based transformers to use contextualized embeddings for Valence/Arousal prediction
- Current state
  Backbone code is completed. We need to automate hyperparamter tuning and test on languages other than English

## /subtask1
Contains model training, eval, logging files

### Files
- **subtask1_model.py**  
  Defines model and tokenized data classes. Has methods for training and evaluating model.

- **subtask1_driver.py**  
  Driver program for training and testing. Takes in command line args for train/test modes.

- **data_loader.py**  
  Helper functions for loading and processing data.

- **root_mse.py**  
  Defines Root MSE loss function class

- **subtask_1_eng_laptop_golds.json**  
  JSONL file of gold data labels

- **subtask_1_eng_laptop_preds.json**  
  JSONL file of VA predictions

- **subtask1_eng_laptop_eval.txt**  
  Output of testing results

- **eval.sh**
  Bash script to run evaluation script

- **metrics_subtask_1_2_3.py**
  Organizer provided eval script

- **/loss_graphs**
  Directory of logged loss data, with graphs in ipynb.
  ODS file has table of model arch. with associated graphs
  
## /subtask1_data
Directory of task dataset

---
### Files
- **st1_eng_laptop.json**  
  JSONL of Subtask 1 English Laptop train/dev/test data

---
