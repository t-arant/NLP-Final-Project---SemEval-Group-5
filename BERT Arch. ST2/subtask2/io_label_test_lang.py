from io_label_gen import *



# testing russian for BIOES classification ---------------------------------------------------------
# restaurant data
#  Russian

# subtask = "subtask_2"
# task = "task1"#don't change
# lang = "rus" #chang the language you want to test
# domain = "restaurant" #change what domain you want to test
# tokenizer = "distilbert-base-multilingual-cased"
# f_name = "subtask_a_rus_rest.json"

# generate_BIOES_labesls(lang, task, subtask, domain, tokenizer, "bioes", f_name)

# aug_df_bioes_lab = df = pd.read_json(f_name, orient="records", lines=True)

# test_labeling(aug_df_bioes_lab, tokenizer, lang, domain)

#----------------------------------------------------------------------------------
# restaurant data
#  Ukrainian

# subtask = "subtask_2"#don't change
# task = "task1"#don't change
# lang = "ukr" #chang the language you want to test
# domain = "restaurant" #change what domain you want to test
# tokenizer = "bert-base-multilingual-cased"
# f_name = "subtask_a_ukr_rest.json"

# generate_BIOES_labesls(lang, task, subtask, domain, tokenizer, "bioes", f_name)

# aug_df_bioes_lab = df = pd.read_json(f_name, orient="records", lines=True)

# test_labeling(aug_df_bioes_lab, tokenizer, lang, domain)


# ------------------------------------------------------------------------------------------------
# restaurant data
#  Tartar

subtask = "subtask_2"#don't change
task = "task1"#don't change
lang = "tat" #chang the language you want to test
domain = "restaurant" #change what domain you want to test
tokenizer = "bert-base-multilingual-cased"
f_name = "subtask_a_tat_rest.json"

generate_BIOES_labesls(lang, task, subtask, domain, tokenizer, "bioes", f_name)

aug_df_bioes_lab = df = pd.read_json(f_name, orient="records", lines=True)

test_labeling(aug_df_bioes_lab, tokenizer, lang, domain)


# ------------------------------------------------------------------------------------------------
# restaurant data
#  Chinese

subtask = "subtask_2"#don't change
task = "task1"#don't change
lang = "zho" #chang the language you want to test
domain = "restaurant" #change what domain you want to test
tokenizer = "bert-base-chinese"
f_name = "subtask_a_zho_rest.json"

generate_BIOES_labesls(lang, task, subtask, domain, tokenizer, "bioes", f_name)

aug_df_bioes_lab = df = pd.read_json(f_name, orient="records", lines=True)

test_labeling(aug_df_bioes_lab, tokenizer, lang, domain)


# ------------------------------------------------------------------------------------------------
# restaurant data
#  Japanese

# subtask = "subtask_1"#don't change
# task = "task1"#don't change
# lang = "jpn" #chang the language you want to test
# domain = "finance" #change what domain you want to test
# tokenizer = "bert-base-multilingual-cased"
# f_name = "subtask_a_jpn_fin.json"

# generate_BIOES_labesls(lang, task, subtask, domain, tokenizer, "bioes", f_name)

# aug_df_bioes_lab = df = pd.read_json(f_name, orient="records", lines=True)

# test_labeling(aug_df_bioes_lab, tokenizer, lang, domain)


# ------------------------------------------------------------------------------------------------
# # restaurant data
# #  English

subtask = "subtask_2"#don't change
task = "task1"#don't change
lang = "eng" #chang the language you want to test
domain = "restaurant" #change what domain you want to test
tokenizer = "bert-base-multilingual-cased"
f_name = "subtask_a_eng_rest.json"

generate_BIOES_labesls(lang, task, subtask, domain, tokenizer, "bioes", f_name)

aug_df_bioes_lab = df = pd.read_json(f_name, orient="records", lines=True)

test_labeling(aug_df_bioes_lab, tokenizer, lang, domain)


# ------------------------------------------------------------------------------------------------
# # laptop data
# #  English

subtask = "subtask_2"#don't change
task = "task1"#don't change
lang = "eng" #chang the language you want to test
domain = "laptop" #change what domain you want to test
tokenizer = "bert-base-multilingual-cased"
f_name = "subtask_a_eng_rest.json"

generate_BIOES_labesls(lang, task, subtask, domain, tokenizer, "bioes", f_name)

aug_df_bioes_lab = df = pd.read_json(f_name, orient="records", lines=True)

test_labeling(aug_df_bioes_lab, tokenizer, lang, domain)

# ------------------------------------------------------------------------------------------------
# laptop data
#  English

subtask = "subtask_2"#don't change
task = "task1"#don't change
lang = "zho" #chang the language you want to test
domain = "laptop" #change what domain you want to test
tokenizer = "bert-base-multilingual-cased"
f_name = "subtask_a_zho_laptop.json"

generate_BIOES_labesls(lang, task, subtask, domain, tokenizer, "bioes", f_name)

aug_df_bioes_lab = df = pd.read_json(f_name, orient="records", lines=True)

test_labeling(aug_df_bioes_lab, tokenizer, lang, domain)