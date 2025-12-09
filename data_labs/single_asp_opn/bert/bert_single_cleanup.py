import pandas as pd 
import json
import ast

# /home/tony-arant/NLP Final Project/data_labs/single_asp_opn/bert/eng_laptop_bert_comb.csv 
# /home/tony-arant/NLP Final Project/data_labs/single_asp_opn/bert/zho_laptop_bert_comb.csv 
# /home/tony-arant/NLP Final Project/data_labs/single_asp_opn/bert/eng_restaurant_bert_comb.csv 
# /home/tony-arant/NLP Final Project/data_labs/single_asp_opn/bert/rus_restaurant_bert_comb.csv 
# /home/tony-arant/NLP Final Project/data_labs/single_asp_opn/bert/tat_restaurant_bert_comb.csv 
# /home/tony-arant/NLP Final Project/data_labs/single_asp_opn/bert/ukr_restaurant_bert_comb.csv 
# /home/tony-arant/NLP Final Project/data_labs/single_asp_opn/bert/zho_restaurant_bert_comb.csv

eng_lap = pd.read_csv("/home/tony-arant/NLP Final Project/data_labs/single_asp_opn/bert/eng_laptop_bert_comb.csv", index_col=0)
eng_rest = pd.read_csv("/home/tony-arant/NLP Final Project/data_labs/single_asp_opn/bert/eng_restaurant_bert_comb.csv", index_col=0)
zho_lap = pd.read_csv("/home/tony-arant/NLP Final Project/data_labs/single_asp_opn/bert/zho_laptop_bert_comb.csv", index_col=0)
zho_rest = pd.read_csv("/home/tony-arant/NLP Final Project/data_labs/single_asp_opn/bert/zho_restaurant_bert_comb.csv", index_col=0)
rus_rest = pd.read_csv("/home/tony-arant/NLP Final Project/data_labs/single_asp_opn/bert/rus_restaurant_bert_comb.csv", index_col=0)
tar_rest = pd.read_csv("/home/tony-arant/NLP Final Project/data_labs/single_asp_opn/bert/tat_restaurant_bert_comb.csv", index_col=0)
ukr_rest = pd.read_csv("/home/tony-arant/NLP Final Project/data_labs/single_asp_opn/bert/ukr_restaurant_bert_comb.csv", index_col=0)


eng_lap["BIOES"] = eng_lap["BIOES"].apply(ast.literal_eval)
# eng_lap["BIOES_opn"] = eng_lap["BIOES_opn"].apply(ast.literal_eval)

eng_rest["BIOES"] = eng_rest["BIOES"].apply(ast.literal_eval)
# eng_rest["BIOES_opn"] = eng_rest["BIOES_opn"].apply(ast.literal_eval)

zho_lap["BIOES"] = zho_lap["BIOES"].apply(ast.literal_eval)
# zho_lap["BIOES_opn"] = zho_lap["BIOES_opn"].apply(ast.literal_eval)

zho_rest["BIOES"] = zho_rest["BIOES"].apply(ast.literal_eval)
# zho_rest["BIOES_opn"] = zho_rest["BIOES_opn"].apply(ast.literal_eval)

rus_rest["BIOES"] = rus_rest["BIOES"].apply(ast.literal_eval)
# rus_rest["BIOES_opn"] = rus_rest["BIOES_opn"].apply(ast.literal_eval)

tar_rest["BIOES"] = tar_rest["BIOES"].apply(ast.literal_eval)
# tar_rest["BIOES_opn"] = tar_rest["BIOES_opn"].apply(ast.literal_eval)

ukr_rest["BIOES"] = ukr_rest["BIOES"].apply(ast.literal_eval)
# ukr_rest["BIOES_opn"] = ukr_rest["BIOES_opn"].apply(ast.literal_eval)

for i in range(len(eng_lap)):
    lab = eng_lap.iloc[i]["BIOES"]

    lab_len = len(lab)
    
    while lab_len < 128:
        lab.append('EMB_P')
        lab_len += 1

for i in range(len(eng_rest)):
    lab = eng_rest.iloc[i]["BIOES"]

    lab_len = len(lab)
    
    while lab_len < 128:
        lab.append('EMB_P')
        lab_len += 1

for i in range(len(zho_lap)):
    lab = zho_lap.iloc[i]["BIOES"]

    lab_len = len(lab)
    
    while lab_len < 512:
        lab.append('EMB_P')
        lab_len += 1

for i in range(len(zho_rest)):
    lab = zho_rest.iloc[i]["BIOES"]

    lab_len = len(lab)
    
    while lab_len < 512:
        lab.append('EMB_P')
        lab_len += 1

for i in range(len(rus_rest)):
    lab = rus_rest.iloc[i]["BIOES"]

    lab_len = len(lab)
    
    while lab_len < 128:
        lab.append('EMB_P')
        lab_len += 1

for i in range(len(tar_rest)):
    lab = tar_rest.iloc[i]["BIOES"]

    lab_len = len(lab)
    
    while lab_len < 128:
        lab.append('EMB_P')
        lab_len += 1

for i in range(len(ukr_rest)):
    lab = ukr_rest.iloc[i]["BIOES"]

    lab_len = len(lab)
    
    while lab_len < 128:
        lab.append('EMB_P')
        lab_len += 1

eng_lap.drop([66,2706,3075,3153,3412,3531,3573,3866,3871,3883,3991,4023,4053,4085], inplace=True)

eng_rest.drop([77,197,1392,1421,1663,1681,1909,2470,2533,2827,2840], inplace=True)

zho_lap.drop([220,300,322,385,513,581,894,934,1059,1142,1232,1380,1423,1466,1507,1539,1638,1679,1777,1792,1796,1799,1849,1854,1861,1907,2029,2042,2178,2234,2249,2297,2303,2498,2558,2569,2933,2934,2950,2959,3093,3121,3133,3154,3197,3269,3386,3429,3494,3542,3618,3797,4001,4205,4335,4336,4354,4377,4384,4395,4416,4434,4453,4557,4686,4811,4898,4930,5045,5105,5280,5410,5438,5453,5489,5553,5597,5657,5739,5810,5860,5876], 
                inplace=True)

zho_rest.drop([88,218,253,279,377,382,389,398,424,465,534,541,564,589,598,668,685,714,789,790,791,823,846,851,870,912,939,987,988,1036,1053,1059,1072,1095,1129,1135,1145,1168,1169,1170,1171,1184,1226,1234,1277,1310,1313,1315,1324,1344,1350,1429,1435,1478,1485,1504,1505,1551,1556,1580,1581,1590,1614,1652,1657,1659,1681,1705,1706,1723,1746,1758,1771,1772,1790,1847,1848,1882,1920,1921,1929,1942,1945,1991,2007,2008,2035,2119,2127,2143,2193,2230,2231,2276,2284,2299,2380,2398,2414,2446,2449,2451,2473,2478,2487,2514,2549,2556,2579,2590,2623,2624,2655,2697,2698,2712,2734,2739,2740,2764,2783,2784,2807,2811,2821,2839,2869,2878,2879,2902,2904,2918,2920,3011,3019,3034,3054,3094,3103,3104,3105,3124,3126,3130,3187,3231,3244,3297,3307,3337,3364,3384,3388,3430,3439,3452,3454,3461,3462,3519,3523,3565,3627,3628,3629,3634,3637,3657,3665,3687,3698,3756,3767,3772,3773,3799,3809,3851,3852,3860,3869,3874,3875,3893,3909,3933,3934,3939,3942,3957,3958,3996,4030,4060,4078,4082,4114,4159,4178,4191,4197,4206,4218,4226,4227,4239,4243,4250,4291,4293,4308,4311,4320,4347,4469,4474,4493,4496,4522,4604,4605,4606,4625,4626,4633,4642,4647,4661,4670,4689,4712,4714,4723,4757,4780,4834,4851,4868,4872,4876,4888,4910,4952,4980,5012,5022,5089,5115,5126,5127,5139,5146,5165,5277,5320,5346,5355,5417,5433,5457,5475,5495,5505,5509,5512,5514,5525,5537,5559,5568,5569,5576,5586,5609,5625,5626,5633,5645,5684,5697,5742,5808,5816,5836,5846,5847,5848,5857,5860,5861,5863,5865,5875,5893,5895,5897,5902,5903,5906,5917,5921,5937,5989,6003,6011,6022,6155,6203,6221,6222,6238,6239,6246,6248,6252,6331,6437,6442,6449,6533,6565,6578,6716,6722,6845,6855,6865,6868,6923,6929,6937,7035,7045,7085,7096,7192,7204,7251,7252,7256,7259,7261,7263,7273,7278,7394], 
               inplace=True)

rus_rest.drop([328,383,451,529,677,948,972,1003,1032,1135,1158,1198,1208,1251,1257,1389,1608,1625,1629,1769,1786,1804,1822,1841,1842], inplace=True)

tar_rest.drop([13,82,238,239,240,241,291,309,327,362,369,381,413,432,452,469,602,646,677,707,752,754,834,923,972,983,1017,1038,1071,1087,1097,1160,1253,1259,1270,1357,1557,1599,1628,1712,1775,1778,1789,1851,1878,1903,1916,1938,1964,1965], inplace=True)

ukr_rest.drop([31,241,257,280,318,328,358,373,444,675,684,969,1000,1015,1132,1155,1186,1248,1254,1268,1306,1386,1626,1631,1783,1801,1818,1870,1900,1957], inplace=True)


eng_lap.to_json("bert_subtask_2_eng_laptop_comb.json", orient="records", lines=True)
eng_rest.to_json("bert_subtask_2_eng_restaurant_comb.json", orient="records", lines=True)
zho_lap.to_json("bert_subtask_2_zho_laptop_comb.json", orient="records", lines=True)
zho_rest.to_json("bert_subtask_2_zho_restaurant_comb.json", orient="records", lines=True)
rus_rest.to_json("bert_subtask_2_rus_restaurant_comb.json", orient="records", lines=True)
tar_rest.to_json("bert_subtask_2_tat_restaurant_comb.json", orient="records", lines=True)
ukr_rest.to_json("bert_subtask_2_ukr_restaurant_comb.json", orient="records", lines=True)

# /home/tony-arant/NLP Final Project/data_labs/single_asp_opn/distilbert/old_json/subtask_2_eng_laptop_comb.json 
# /home/tony-arant/NLP Final Project/data_labs/single_asp_opn/distilbert/old_json/subtask_2_eng_restaurant_comb.json 
# /home/tony-arant/NLP Final Project/data_labs/single_asp_opn/distilbert/old_json/subtask_2_rus_restaurant_comb.json 
# /home/tony-arant/NLP Final Project/data_labs/single_asp_opn/distilbert/old_json/subtask_2_tat_restaurant_comb.json 
# /home/tony-arant/NLP Final Project/data_labs/single_asp_opn/distilbert/old_json/subtask_2_ukr_restaurant_comb.json 
# /home/tony-arant/NLP Final Project/data_labs/single_asp_opn/distilbert/old_json/subtask_2_zho_laptop_comb.json 
# /home/tony-arant/NLP Final Project/data_labs/single_asp_opn/distilbert/old_json/subtask_2_zho_restaurant_comb.json