#!/bin/bash

python metrics_subtask_1_2_3.py -t 2 \
  -p "comb/bert_subtask_2_eng_lapt_preds_crf_comb.json" \
  -g "comb/bert_subtask_2_eng_lapt_golds.json" > "c_bert_eng_lapt_res.txt"

python metrics_subtask_1_2_3.py -t 2 \
  -p "comb/bert_subtask_2_eng_rest_preds_crf_comb.json" \
  -g "comb/bert_subtask_2_eng_rest_golds.json" > "c_bert_eng_rest_res.txt"

python metrics_subtask_1_2_3.py -t 2 \
  -p "comb/bert_subtask_2_rus_rest_preds_crf_comb.json" \
  -g "comb/bert_subtask_2_rus_rest_golds.json" > "c_bert_rus_rest_res.txt"

python metrics_subtask_1_2_3.py -t 2 \
  -p "comb/bert_subtask_2_tat_rest_preds_crf_comb.json" \
  -g "comb/bert_subtask_2_tat_rest_golds.json" > "c_bert_tat_rest_res.txt"

python metrics_subtask_1_2_3.py -t 2 \
  -p "comb/bert_subtask_2_ukr_rest_preds_crf_comb.json" \
  -g "comb/bert_subtask_2_ukr_rest_golds.json" > "c_bert_ukr_rest_res.txt"

python metrics_subtask_1_2_3.py -t 2 \
  -p "comb/bert_subtask_2_zho_lapt_preds_crf_comb.json" \
  -g "comb/bert_subtask_2_zho_lapt_golds.json" > "c_bert_zho_lapt_res.txt"

python metrics_subtask_1_2_3.py -t 2 \
  -p "comb/bert_subtask_2_zho_rest_preds_crf_comb.json" \
  -g "comb/bert_subtask_2_zho_rest_golds.json" > "c_bert_zho_rest_res.txt"





python metrics_subtask_1_2_3.py -t 2 \
  -p "comb/distilbert_subtask_2_eng_lapt_preds_crf_comb.json" \
  -g "comb/distilbertbert_subtask_2_eng_lapt_golds.json" > "c_distilbert_eng_lapt_res.txt"

python metrics_subtask_1_2_3.py -t 2 \
  -p "comb/distililbert_subtask_2_eng_rest_preds_crf_comb.json" \
  -g "comb/distililbert_subtask_2_eng_rest_golds.json" > "c_distilbert_eng_lapt_res.txt"

python metrics_subtask_1_2_3.py -t 2 \
  -p "comb/distilbert_subtask_2_rus_rest_preds_crf_comb.json" \
  -g "comb/distilbert_subtask_2_rus_rest_golds.json" > "c_distilbert_rus_rest_res.txt"

python metrics_subtask_1_2_3.py -t 2 \
  -p "comb/distilbert_subtask_2_tat_rest_preds_crf_comb.json" \
  -g "comb/distilbert_subtask_2_tat_rest_golds.json" > "c_distilbert_tat_rest_res.txt"

python metrics_subtask_1_2_3.py -t 2 \
  -p "comb/distilbert_subtask_2_ukr_rest_preds_crf_comb.json" \
  -g "comb/distilbert_subtask_2_ukr_rest_golds.json" > "c_distilbert_ukr_rest_res.txt"

python metrics_subtask_1_2_3.py -t 2 \
  -p "comb/distilbert_subtask_2_zho_lapt_preds_crf_comb.json" \
  -g "comb/distilbertbert_subtask_2_zho_lapt_golds.json" > "c_distilbert_zho_lapt_res.txt"

python metrics_subtask_1_2_3.py -t 2 \
  -p "comb/distilbert_subtask_2_zho_rest_preds_crf_comb.json" \
  -g "comb/distilbertbert_subtask_2_zho_rest_golds.json" > "c_distilbert_zho_rest_res.txt"


  # comb/distilbertbert_subtask_2_eng_lapt_golds.json 

  # comb/distilbertbert_subtask_2_zho_lapt_golds.json 

  # comb/distilbertbert_subtask_2_zho_rest_golds.json 

  # comb/distilbert_subtask_2_eng_lapt_preds_crf_comb.json 

  # comb/distilbert_subtask_2_rus_rest_golds.json 

  # comb/distilbert_subtask_2_rus_rest_preds_crf_comb.json 

  # comb/distilbert_subtask_2_tat_rest_golds.json 

  # comb/distilbert_subtask_2_tat_rest_preds_crf_comb.json 

  # comb/distilbert_subtask_2_ukr_rest_golds.json 

  # comb/distilbert_subtask_2_ukr_rest_preds_crf_comb.json 

  # comb/distilbert_subtask_2_zho_lapt_preds_crf_comb.json 

  # comb/distilbert_subtask_2_zho_rest_preds_crf_comb.json 

  # comb/distililbert_subtask_2_eng_rest_golds.json 
  # comb/distililbert_subtask_2_eng_rest_preds_crf_comb.json