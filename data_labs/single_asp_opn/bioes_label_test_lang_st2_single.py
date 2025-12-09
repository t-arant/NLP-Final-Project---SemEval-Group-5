from bioes_label_gen_st2_single import *



langs = ["eng", "zho", "ukr", "rus", "tat"]

domain1 = ["laptop", "restaurant"]
domain2 = ["restaurant"]
tknr_list = ["distilbert-base-multilingual-cased", "bert-base-multilingual-cased"]

tkr = "distilbert-base-multilingual-cased"
for lang in langs:

        if (lang == "eng") or (lang == "zho"):
            z = 1

            for domain in domain1: 
                subtask = "subtask_2"#don't change
                task = "task1"#don't change
                # lang = "tat" #chang the language you want to test
                # domain = "restaurant" #change what domain you want to test
                # tokenizer = "bert-base-multilingual-cased"
                f_name = f"subtask_2_{lang}_{domain}_comb.json"

                if lang == "zho":
                    max_len = 512
                else: 
                    max_len = 128

                generate_BIOES_labesls(lang, task, subtask, domain, tkr, "bioes", f_name, max_len)

                aug_df_bioes_lab = df = pd.read_json(f_name, orient="records", lines=True)

                if tkr == "distilbert-base-multilingual-cased":
                    tok_type = "distilbert"
                elif tkr == "bert-base-multilingual-cased":    
                    tok_type = "bert"

                elif tkr == "xlm-roberta-base":    
                    tok_type = "roberta"

                test_labeling(aug_df_bioes_lab, tkr, lang, domain, max_len, tok_type)

        else: 
            for domain in domain2: 
                subtask = "subtask_2"#don't change
                task = "task1"#don't change
                # lang = "tat" #chang the language you want to test
                # domain = "restaurant" #change what domain you want to test
                # tokenizer = "bert-base-multilingual-cased"
                f_name = f"subtask_2_{lang}_{domain}_comb.json"

                max_len = 128

                generate_BIOES_labesls(lang, task, subtask, domain, tkr, "bioes", f_name, max_len)

                aug_df_bioes_lab = df = pd.read_json(f_name, orient="records", lines=True)

                if tkr == "distilbert-base-multilingual-cased":
                    tok_type = "distilbert"
                elif tkr == "bert-base-multilingual-cased":    
                    tok_type = "bert"

                elif tkr == "xlm-roberta-base":    
                    tok_type = "roberta"

                test_labeling(aug_df_bioes_lab, tkr, lang, domain, max_len, tok_type)
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
# task = "task1"#don't changetkrzer
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

# subtask = "subtask_2"#don't change
# task = "task1"#don't change
# lang = "tat" #chang the language you want to test
# domain = "restaurant" #change what domain you want to test
# tokenizer = "bert-base-multilingual-cased"
# f_name = "subtask_a_tat_rest.json"

# generate_BIOES_labesls(lang, task, subtask, domain, tokenizer, "bioes", f_name)

# aug_df_bioes_lab = df = pd.read_json(f_name, orient="records", lines=True)

# test_labeling(aug_df_bioes_lab, tokenizer, lang, domain)


# ------------------------------------------------------------------------------------------------
# restaurant data
#  Chinese

# subtask = "subtask_2"#don't change
# task = "task1"#don't change
# lang = "zho" #chang the language you want to test
# domain = "restaurant" #change what domain you want to test
# tokenizer = "bert-base-chinese"
# f_name = "subtask_a_zho_rest.json"

# generate_BIOES_labesls(lang, task, subtask, domain, tokenizer, "bioes", f_name)

# aug_df_bioes_lab = df = pd.read_json(f_name, orient="records", lines=True)

# test_labeling(aug_df_bioes_lab, tokenizer, lang, domain)


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

# subtask = "subtask_2"#don't change
# task = "task1"#don't change
# lang = "eng" #chang the language you want to test
# domain = "restaurant" #change what domain you want to test
# tokenizer = "bert-base-multilingual-cased"
# f_name = "subtask_a_eng_rest.json"

# generate_BIOES_labesls(lang, task, subtask, domain, tokenizer, "bioes", f_name)

# aug_df_bioes_lab = df = pd.read_json(f_name, orient="records", lines=True)

# test_labeling(aug_df_bioes_lab, tokenizer, lang, domain)


# ------------------------------------------------------------------------------------------------
# # laptop data
# #  English

# subtask = "subtask_2"#don't change
# task = "task1"#don't change
# lang = "eng" #chang the language you want to test
# domain = "laptop" #change what domain you want to test
# tokenizer = "bert-base-multilingual-cased"
# f_name = "subtask_a_eng_rest.json"

# generate_BIOES_labesls(lang, task, subtask, domain, tokenizer, "bioes", f_name)

# aug_df_bioes_lab = df = pd.read_json(f_name, orient="records", lines=True)

# test_labeling(aug_df_bioes_lab, tokenizer, lang, domain)

# ------------------------------------------------------------------------------------------------
# laptop data
#  English

# subtask = "subtask_2"#don't change
# task = "task1"#don't change
# lang = "zho" #chang the language you want to test
# domain = "laptop" #change what domain you want to test
# tokenizer = "bert-base-multilingual-cased"
# f_name = "subtask_a_zho_laptop.json"

# generate_BIOES_labesls(lang, task, subtask, domain, tokenizer, "bioes", f_name)

# aug_df_bioes_lab = df = pd.read_json(f_name, orient="records", lines=True)

# test_labeling(aug_df_bioes_lab, tokenizer, lang, domain)