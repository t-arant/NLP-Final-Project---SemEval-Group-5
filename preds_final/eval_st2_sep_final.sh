#!/bin/bash

python metrics_subtask_1_2_3.py -t 2 \
  -p "sep/subtask_2_eng_lapt_preds_crf_sep_bert.json" \
  -g "sep/subtask_2_eng_lapt_golds_crf_sep_bert.json" > "bert_eng_lapt_res.txt"

python metrics_subtask_1_2_3.py -t 2 \
  -p "sep/subtask_2_eng_lapt_preds_crf_sep_distilbert.json" \
  -g "sep/subtask_2_eng_lapt_golds_crf_sep_distilbert.json" > "distlbert_eng_lapt_res.txt"


python metrics_subtask_1_2_3.py -t 2 \
  -p "sep/subtask_2_eng_rest_preds_crf_sep_bert.json"  \
  -g "sep/subtask_2_eng_rest_golds_crf_sep_bert.json" > "bert_eng_rest_res.txt"

python metrics_subtask_1_2_3.py -t 2 \
  -p "sep/subtask_2_eng_rest_preds_crf_sep_distilbert.json"  \
  -g "sep/subtask_2_eng_rest_golds_crf_sep_distilbert.json" > "distilbert_eng_rest_res.txt"


python metrics_subtask_1_2_3.py -t 2 \
  -p "sep/subtask_2_rus_rest_preds_crf_sep_bert.json"  \
  -g "sep/subtask_2_rus_rest_golds_crf_sep_bert.json" > "bert_rus_rest_res.txt"

python metrics_subtask_1_2_3.py -t 2 \
  -p "sep/subtask_2_rus_rest_preds_crf_sep_distilbert.json"  \
  -g "sep/subtask_2_rus_rest_golds_crf_sep_distilbert.json" > "distilbert_rus_rest_res.txt"



python metrics_subtask_1_2_3.py -t 2 \
  -p "sep/subtask_2_tat_rest_preds_crf_sep_bert.json"  \
  -g "sep/subtask_2_tat_rest_golds_crf_sep_bert.json" > "bert_tat_rest_res.txt"

python metrics_subtask_1_2_3.py -t 2 \
  -p "sep/subtask_2_tat_rest_preds_crf_sep_distilbert.json"  \
  -g "sep/subtask_2_tat_rest_golds_crf_sep_distilbert.json" > "distilbert_tat_rest_res.txt"


python metrics_subtask_1_2_3.py -t 2 \
  -p "sep/subtask_2_ukr_rest_preds_crf_sep_bert.json"  \
  -g "sep/subtask_2_ukr_rest_golds_crf_sep_bert.json" > "bert_ukr_rest_res.txt"

python metrics_subtask_1_2_3.py -t 2 \
  -p "sep/subtask_2_ukr_rest_preds_crf_sep_distilbert.json"  \
  -g "sep/subtask_2_ukr_rest_golds_crf_sep_distilbert.json" > "distilbert_ukr_rest_res.txt"


python metrics_subtask_1_2_3.py -t 2 \
  -p "sep/subtask_2_zho_lapt_preds_crf_sep_bert.json"  \
  -g "sep/subtask_2_zho_lapt_golds_crf_sep_bert.json" > "bert_zho_lapt_res.txt"

python metrics_subtask_1_2_3.py -t 2 \
  -p "sep/subtask_2_zho_lapt_preds_crf_sep_distilbert.json"  \
  -g "sep/subtask_2_zho_lapt_golds_crf_sep_distilbert.json" > "distilbert_zho_lapt_res.txt"


python metrics_subtask_1_2_3.py -t 2 \
  -p "sep/subtask_2_zho_rest_preds_crf_sep_bert.json"  \
  -g "sep/subtask_2_zho_rest_golds_crf_sep_bert.json" > "bert_zho_rest_res.txt"

python metrics_subtask_1_2_3.py -t 2 \
  -p "sep/subtask_2_zho_rest_preds_crf_sep_distilbert.json"  \
  -g "sep/subtask_2_zho_rest_golds_crf_sep_distilbert.json" > "distilbert_zho_rest_res.txt"


