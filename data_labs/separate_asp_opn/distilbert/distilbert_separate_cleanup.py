import pandas as pd 
import json
import ast 

# /home/tony-arant/NLP Final Project/data_labs/separate_asp_opn/distilbert/eng_laptop_distilbert_sep.csv 1
# /home/tony-arant/NLP Final Project/data_labs/separate_asp_opn/distilbert/zho_laptop_distilbert_sep.csv 1
# /home/tony-arant/NLP Final Project/data_labs/separate_asp_opn/distilbert/eng_restaurant_distilbert_sep.csv 1 
# /home/tony-arant/NLP Final Project/data_labs/separate_asp_opn/distilbert/rus_restaurant_distilbert_sep.csv 1
# /home/tony-arant/NLP Final Project/data_labs/separate_asp_opn/distilbert/tat_restaurant_distilbert_sep.csv 1
# /home/tony-arant/NLP Final Project/data_labs/separate_asp_opn/distilbert/ukr_restaurant_distilbert_sep.csv 
# /home/tony-arant/NLP Final Project/data_labs/separate_asp_opn/distilbert/zho_restaurant_distilbert_sep.csv 1

eng_lap = pd.read_csv("/home/tony-arant/NLP Final Project/data_labs/separate_asp_opn/distilbert/eng_laptop_distilbert_sep.csv", index_col=0)
eng_rest = pd.read_csv("/home/tony-arant/NLP Final Project/data_labs/separate_asp_opn/distilbert/eng_restaurant_distilbert_sep.csv", index_col=0)
zho_lap = pd.read_csv("/home/tony-arant/NLP Final Project/data_labs/separate_asp_opn/distilbert/zho_laptop_distilbert_sep.csv", index_col=0)
zho_rest = pd.read_csv("/home/tony-arant/NLP Final Project/data_labs/separate_asp_opn/distilbert/zho_restaurant_distilbert_sep.csv", index_col=0)
rus_rest = pd.read_csv("/home/tony-arant/NLP Final Project/data_labs/separate_asp_opn/distilbert/rus_restaurant_distilbert_sep.csv", index_col=0)
tar_rest = pd.read_csv("/home/tony-arant/NLP Final Project/data_labs/separate_asp_opn/distilbert/tat_restaurant_distilbert_sep.csv", index_col=0)
ukr_rest = pd.read_csv("/home/tony-arant/NLP Final Project/data_labs/separate_asp_opn/distilbert/ukr_restaurant_distilbert_sep.csv", index_col=0)

eng_lap["BIOES_asp"] = eng_lap["BIOES_asp"].apply(ast.literal_eval)
eng_lap["BIOES_opn"] = eng_lap["BIOES_opn"].apply(ast.literal_eval)

eng_rest["BIOES_asp"] = eng_rest["BIOES_asp"].apply(ast.literal_eval)
eng_rest["BIOES_opn"] = eng_rest["BIOES_opn"].apply(ast.literal_eval)

zho_lap["BIOES_asp"] = zho_lap["BIOES_asp"].apply(ast.literal_eval)
zho_lap["BIOES_opn"] = zho_lap["BIOES_opn"].apply(ast.literal_eval)

zho_rest["BIOES_asp"] = zho_rest["BIOES_asp"].apply(ast.literal_eval)
zho_rest["BIOES_opn"] = zho_rest["BIOES_opn"].apply(ast.literal_eval)

rus_rest["BIOES_asp"] = rus_rest["BIOES_asp"].apply(ast.literal_eval)
rus_rest["BIOES_opn"] = rus_rest["BIOES_opn"].apply(ast.literal_eval)

tar_rest["BIOES_asp"] = tar_rest["BIOES_asp"].apply(ast.literal_eval)
tar_rest["BIOES_opn"] = tar_rest["BIOES_opn"].apply(ast.literal_eval)

ukr_rest["BIOES_asp"] = ukr_rest["BIOES_asp"].apply(ast.literal_eval)
ukr_rest["BIOES_opn"] = ukr_rest["BIOES_opn"].apply(ast.literal_eval)


rus_rest["BIOES_asp"] = rus_rest["BIOES_asp"].apply(
    lambda lst: lst[:128] if isinstance(lst, list) and len(lst) > 128 else lst
)

rus_rest["BIOES_opn"] = rus_rest["BIOES_opn"].apply(
    lambda lst: lst[:128] if isinstance(lst, list) and len(lst) > 128 else lst
)


ukr_rest["BIOES_asp"] = ukr_rest["BIOES_asp"].apply(
    lambda lst: lst[:128] if isinstance(lst, list) and len(lst) > 128 else lst
)

ukr_rest["BIOES_opn"] = ukr_rest["BIOES_opn"].apply(
    lambda lst: lst[:128] if isinstance(lst, list) and len(lst) > 128 else lst
)

tar_rest["BIOES_asp"] = tar_rest["BIOES_asp"].apply(
    lambda lst: lst[:128] if isinstance(lst, list) and len(lst) > 128 else lst
)

tar_rest["BIOES_opn"] = tar_rest["BIOES_opn"].apply(
    lambda lst: lst[:128] if isinstance(lst, list) and len(lst) > 128 else lst
)

flag = 0
print("eng_lap_asp")
for x in range(len(eng_lap)):
    if len(eng_lap.iloc[x]["BIOES_asp"]) != 128:
        flag = 1

if flag == 1: 
    print("prob")

flag = 0

print("eng_lap_opn")
for x in range(len(eng_lap)):
    if len(eng_lap.iloc[x]["BIOES_asp"]) != 128:
        flag = 1
if flag == 1: 
    print("prob")
flag = 0

print("eng_rest_asp")
for x in range(len(eng_rest)):
    if len(eng_rest.iloc[x]["BIOES_asp"]) != 128:
        flag = 1
if flag == 1: 
    print("prob")
flag = 0

print("eng_rest_opn")
for x in range(len(eng_rest)):
    if len(eng_rest.iloc[x]["BIOES_asp"]) != 128:
        flag = 1
if flag == 1: 
    print("prob")
flag = 0

print("rus_rest_asp")
for x in range(len(rus_rest)):
    if len(rus_rest.iloc[x]["BIOES_asp"]) != 128:
        flag = 1
if flag == 1: 
    print("prob")
flag = 0

print("rus_rest_opn")
for x in range(len(rus_rest)):
    if len(rus_rest.iloc[x]["BIOES_asp"]) != 128:
        flag = 1
if flag == 1: 
    print("prob")
flag = 0

print("ukr_rest_asp")
for x in range(len(ukr_rest)):
    if len(ukr_rest.iloc[x]["BIOES_asp"]) != 128:
        flag = 1
if flag == 1: 
    print("prob")
flag = 0

print("ukr_rest_opn")
for x in range(len(ukr_rest)):
    if len(ukr_rest.iloc[x]["BIOES_asp"]) != 128:
        flag = 1

if flag == 1: 
    print("prob")
flag = 0

print("tat_rest_asp")
for x in range(len(tar_rest)):
    if len(tar_rest.iloc[x]["BIOES_asp"]) != 128:
        flag = 1
if flag == 1: 
    print("prob")
flag = 0

print("tat_rest_opn")
for x in range(len(tar_rest)):
    if len(tar_rest.iloc[x]["BIOES_asp"]) != 128:
        flag = 1

if flag == 1: 
    print("prob")
flag = 0

print("zho_lap_asp")
for x in range(len(zho_lap)):
    if len(zho_lap.iloc[x]["BIOES_asp"]) != 512:
        flag = 1
if flag == 1: 
    print("prob")
flag = 0

print("zho_lap_opn")
for x in range(len(zho_lap)):
    if len(zho_lap.iloc[x]["BIOES_asp"]) != 512:
        flag = 1
if flag == 1: 
    print("prob")
flag = 0

print("zho_rest_asp")
for x in range(len(zho_rest)):
    if len(zho_rest.iloc[x]["BIOES_asp"]) != 512:
        flag = 1
if flag == 1: 
    print("prob")
flag = 0

print("zho_rest_opn")
for x in range(len(zho_rest)):
    if len(zho_rest.iloc[x]["BIOES_asp"]) != 512:
        flag = 1

eng_lap.drop([66,847,3075,3573,3991], inplace=True)

eng_rest.drop([197,703,704,1663,1681,1909,2470,2533,2827,2840], inplace=True)

zho_lap.drop([150,385,513,894,934,1142,1232,1638,1679,1743,1777,1796,1849,2029,2042,2178,2234,2249,2297,2303,2498,2500,2558,2569,2740,2933,2934,2950,2959,
              3093,3121,3133,3154,3197,3269,3386,3429,3542,3618,3797,3990,4205,4336,4354,4377,4384,4416,4434,4453,4686,4898,5045,5280,5410,5438,5453,5489,5553,5657,5739,5810,5876], 
              inplace=True)

zho_rest.drop([88,218,253,279,377,382,389,398,424,534,541,564,589,598,668,685,714,789,790,791,823,846,851,870,912,987,988,1036,1053,1059,1072,1095,1129,1145,1168,1169,1170,1171,1184,1226,
               1234,1277,1310,1313,1315,1324,1344,1350,1429,1435,1485,1504,1505,1551,1556,1580,1581,1590,1614,1652,1657,1659,1705,1706,1723,1758,1771,1772,1790,1847,1848,1882,1920,1921,1929,
               1942,1945,1991,2007,2008,2035,2119,2127,2143,2193,2230,2231,2276,2284,2299,2380,2414,2446,2449,2451,2473,2478,2514,2549,2556,2579,2590,2623,2624,2655,2697,2698,2712,2734,2739,2740,
               2764,2783,2784,2807,2811,2821,2839,2869,2878,2879,2902,2904,2918,2920,3011,3019,3034,3054,3094,3103,3104,3105,3124,3126,3130,3187,3231,3307,3337,3364,3384,3388,3430,3439,3452,3454,3461,
               3462,3519,3523,3565,3627,3628,3629,3634,3637,3657,3665,3687,3698,3756,3767,3772,3773,3799,3809,3851,3852,3860,3869,3874,3875,3893,3909,3933,3934,3939,3942,3957,3958,3996,4060,4078,4082,
               4114,4159,4178,4191,4197,4206,4218,4226,4227,4239,4243,4250,4291,4293,4308,4311,4320,4347,4469,4474,4493,4522,4604,4605,4606,4625,4626,4642,4647,4661,4670,4689,4712,4714,4723,4757,4780,
               4834,4868,4872,4876,4888,4910,4952,4980,5012,5022,5089,5115,5126,5127,5139,5146,5165,5277,5320,5346,5355,5417,5433,5457,5475,5495,5505,5509,5512,5514,5525,5537,5559,5568,5569,5576,5625,
               5626,5633,5645,5697,5742,5808,5816,5846,5847,5848,5857,5860,
               5861,5863,5865,5875,5893,5895,5897,5902,5903,5906,5917,5937,5989,6011,6022,6155,6203,6221,6238,6239,
               6246,6248,6252,6331,6437,6442,6449,6533,6565,6578,6716,6722,6845,6855,6865,6868,6923,6929,6937,7035,7045,7085,7096,7192,7204,7251,7252,7256,7259,7261,7263,7273,7278,7394], 
               inplace=True)

rus_rest.drop([328,383,451,677,948,972,1158,1198,1208,1389,1608,1786,1841,1842], inplace=True)

tar_rest.drop([238,309,327,369,432,646,677,707,972,1017,1160,1789,1851,1903,1916], inplace=True)

ukr_rest.drop([257,280,328,675,969,1015,1155,1386,1631,1783,1870,1900], inplace=True)


eng_lap.to_json("distilbert_subtask_2_eng_laptop_sep.json", orient="records", lines=True)
eng_rest.to_json("distilbert_subtask_2_eng_restaurant_sep.json", orient="records", lines=True)
zho_lap.to_json("distilbert_subtask_2_zho_laptop_sep.json", orient="records", lines=True)
zho_rest.to_json("distilbert_subtask_2_zho_restaurant_sep.json", orient="records", lines=True)
rus_rest.to_json("distilbert_subtask_2_rus_restaurant_sep.json", orient="records", lines=True)
tar_rest.to_json("distilbert_subtask_2_tat_restaurant_sep.json", orient="records", lines=True)
ukr_rest.to_json("distilbert_subtask_2_ukr_restaurant_sep.json", orient="records", lines=True)

# /home/tony-arant/NLP Final Project/data_labs/separate_asp_opn/distilbert/subtask_2_eng_laptop_sep.json 
# /home/tony-arant/NLP Final Project/data_labs/separate_asp_opn/distilbert/subtask_2_zho_laptop_sep.json 
# /home/tony-arant/NLP Final Project/data_labs/separate_asp_opn/distilbert/subtask_2_eng_restaurant_sep.json 
# /home/tony-arant/NLP Final Project/data_labs/separate_asp_opn/distilbert/subtask_2_rus_restaurant_sep.json 
# /home/tony-arant/NLP Final Project/data_labs/separate_asp_opn/distilbert/subtask_2_tat_restaurant_sep.json 
# /home/tony-arant/NLP Final Project/data_labs/separate_asp_opn/distilbert/subtask_2_ukr_restaurant_sep.json 
# /home/tony-arant/NLP Final Project/data_labs/separate_asp_opn/distilbert/subtask_2_zho_restaurant_sep.json