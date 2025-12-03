from subtask1_data_genarator import *



# testing russian for BIOES classification ---------------------------------------------------------
# restaurant data
#  Russian

subtask = "subtask_1"#don't change
task = "task1"#don't change
lang = "rus" #chang the language you want to test
domain = "restaurant" #change what domain you want to test
tokenizer = "distilbert-base-multilingual-cased"
f_name = "subtask_1_rus_rest.json"

generate_train_data(lang, task, subtask, domain, f_name)

#----------------------------------------------------------------------------------
# restaurant data
#  Ukrainian

subtask = "subtask_1"#don't change
task = "task1"#don't change
lang = "ukr" #chang the language you want to test
domain = "restaurant" #change what domain you want to test
tokenizer = "bert-base-multilingual-cased"
f_name = "subtask_1_ukr_rest.json"

generate_train_data(lang, task, subtask, domain, f_name)



# ------------------------------------------------------------------------------------------------
# restaurant data
#  Tartar

subtask = "subtask_1"#don't change
task = "task1"#don't change
lang = "tat" #chang the language you want to test
domain = "restaurant" #change what domain you want to test
tokenizer = "bert-base-multilingual-cased"
f_name = "subtask_1_tat_rest.json"

generate_train_data(lang, task, subtask, domain, f_name)



# ------------------------------------------------------------------------------------------------
# restaurant data
#  Chinese

subtask = "subtask_1"#don't change
task = "task1"#don't change
lang = "zho" #chang the language you want to test
domain = "restaurant" #change what domain you want to test
tokenizer = "bert-base-chinese"
f_name = "subtask_1_zho_rest.json"

generate_train_data(lang, task, subtask, domain, f_name)



# ------------------------------------------------------------------------------------------------
# restaurant data
#  Japanese

subtask = "subtask_1"#don't change
task = "task1"#don't change
lang = "jpn" #chang the language you want to test
domain = "finance" #change what domain you want to test
tokenizer = "bert-base-multilingual-cased"
f_name = "subtask_1_jpn_fin.json"

generate_train_data(lang, task, subtask, domain, f_name)


# ------------------------------------------------------------------------------------------------
# # restaurant data
# #  English

subtask = "subtask_1"#don't change
task = "task1"#don't change
lang = "eng" #chang the language you want to test
domain = "restaurant" #change what domain you want to test
tokenizer = "bert-base-multilingual-cased"
f_name = "subtask_1_eng_rest.json"

generate_train_data(lang, task, subtask, domain, f_name)


# ------------------------------------------------------------------------------------------------
# laptop data
#  English

subtask = "subtask_1"#don't change
task = "task1"#don't change
lang = "zho" #chang the language you want to test
domain = "laptop" #change what domain you want to test
tokenizer = "bert-base-multilingual-cased"
f_name = "subtask_1_zho_laptop.json"

generate_train_data(lang, task, subtask, domain, f_name)

