# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:21:24 2026

@author: david
"""

#%%
import matplotlib.pyplot as plt
import numpy as np

#Svi s trenirani do 700000
pocetni_model_win_3_neg_64 = [
522.867859,115.348961,75.979523,63.832275,26.372711,45.360073,38.129669,20.127237,23.959736,41.102520,
34.120697,34.005173,16.656837,28.772411,24.968479,11.129095,23.043701,12.992799,21.411161,10.553698,
17.254187,14.487522,17.156342,14.962017,7.235065,9.201318,11.889511,12.716638,6.525376,9.025875,10.759938,
11.912977,9.908773,10.464991,13.597948,8.448412,7.900062,8.077613,7.557006,8.625727,7.673464,18.288294,
11.204226,7.381090,6.452187,12.345102,5.724663,21.745663,6.446051,10.843539,9.412798,9.763112,9.881784,
8.034800,14.108096,7.707937,5.868593,8.076571,6.840003,5.886861,6.662843,6.425797,5.867273,8.418879,
6.894944,7.465242,6.008080,4.882342,5.606300,7.392670]


model_konekst_win_4_neg_64 = [
520.8893,97.3462,85.9025,50.7843,26.5208,45.8551,38.8879,20.7405,36.4423,84.5965,26.3600,    
29.0817,18.6196,39.3564,25.3487,12.3267,17.9464,14.8679,24.9725,9.9723,19.2798,
11.7295,12.2042,15.5485,7.9384,12.3233,11.5754,9.2514,6.9075,9.7955,11.6539,
14.1416,12.1814,8.9413,12.9283,11.9449,7.2237,8.5384,6.9152,8.4255,9.2754,
6.1760,28.5732,10.9880,9.0814,6.8131,7.6341,7.0754,16.3003,6.4127,10.3940,
8.2278,9.4927,9.6104,6.4422,10.8887,7.3148,5.9381,7.6637,6.3053,6.7211,
4.2174,5.9453,7.1932,9.5952,6.9481,8.8142,7.3304,4.9594,6.9659]

model_kontekst_win_5_neg_64 = [
511.6216,112.9025,68.4929,70.7407,30.1030,59.8352,45.0728,15.5141,26.2048,62.2293,
36.0966,42.8860,17.5817,24.3531,24.2615,14.8234,20.3964,12.3748,18.2359,9.1279,20.7719,
13.2809,14.5575,16.5625,7.7451,10.3092,12.2921,8.2589,6.4148,10.2103,10.1351,17.7367,
10.3631,10.3665,9.2533,14.5616,7.7334,8.0833,7.6473,8.9120,10.0718,7.8177,13.8018,
12.5263,6.8558,6.7665,11.3261,5.1899,23.6138,6.5186,11.5422,11.3519,9.1170,10.6724,
7.2191,10.9127,7.1586,5.4555,6.6672,5.7397,7.4970,4.9312,6.0482,7.8386,8.8196,8.3844,
6.8641,6.8255,5.0288,6.1619]


#ovaj treniran do 1_000_000
model_kontekst_3_neg_128 = [
1010.5344,
103.7940,68.3328,75.4784,43.7859,42.9201,19.7757,15.7926,127.8613,20.2314,13.8518,
29.5716,24.4944,24.7949,22.6673,10.1036,19.7042,12.4126,20.7971,15.2841,20.5217,
13.2124,11.0670,11.3438,13.0676,15.6226,14.7058,10.7300,10.0690,10.0157,12.0159,
10.5939,7.0608,9.9634,9.6730,8.1829,9.4070,10.3750,6.8530,6.5970,9.6427,
6.0691,6.7277,21.1192,8.1329,9.9689,7.9432,9.2685,8.4023,8.4537,9.6369,
6.6431,9.1655,5.6445,8.1283,8.3964,10.5477,9.3006,6.4188,6.3800,4.9423,
7.5593,8.0396,8.0953,6.2714,8.0916,14.8490,6.8483,9.6535,8.0788,8.9632,
8.4328,7.3605,9.2972,6.4867,10.2997,5.3783,7.8156,6.1436,6.8290,6.6502,
8.0492,7.6861,7.2495,10.3651,5.8653,6.1611,6.8489,6.9256,6.3825,6.0875,
7.6068,6.1523,7.2916,8.5309,6.0630,7.0656,5.3575,6.1614,5.9825,6.9470    
]


model_kontekst_4_neg_128 = [
1005.9958,97.1378,68.6425,59.6542,21.4683,37.9839,53.3320,15.6025,35.7236,81.3333,
35.7829,27.8292,20.8489,28.4538,26.0148,10.8730,16.7042,13.7268,22.1086,8.2124,
20.7484,12.4667,18.1744,12.8122,7.4998,9.6156,9.5331,9.7268,7.3448,10.0791,
8.9381,13.0834,11.1338,9.3686,11.0683,13.9420,8.9085,6.9065,7.5313,6.8374,
10.1276,6.8504,14.3248,9.5791,8.9345,6.5723,11.6557,6.4908,17.6904,7.0303,
10.4301,9.5638,7.8443,8.9303,8.2580,10.3973,7.2011,6.0098,8.3059,7.0039,
7.0684,4.8275,6.2560,7.5240,7.9965,7.5619,8.2079,7.1467,5.2472,7.7878,7.1087]

model_kontekst_5_neg_128 =[
1051.1777,
134.7964,69.2589,59.2727,28.6427,53.0154,48.6368,14.8204,33.0524,71.0029,33.4723,
26.4420,14.2381,45.8125,23.3770,10.7221,20.3143,12.0433,19.1103,9.3521,18.8798,
8.4271,13.5557,13.9832,7.1242,11.2635,10.6340,9.4169,7.4577,8.3399,9.7743,
11.9137,9.5850,8.0005,8.9750,14.3753,7.9682,7.5956,7.7884,7.2537,9.5274,
6.7301,17.3236,9.9711,7.2289,6.8043,10.3985,6.4823,19.0908,6.5971,12.5444,
9.2147,8.5724,9.5355,8.0567,11.6720,8.1780,6.1557,7.3013,6.7627,6.5058 ,
4.9887,6.7682,10.3093,8.9583,7.7592,7.8321,6.9422,5.4041,7.0585,6.8891   
]

pocetak_model_kont_3_neg_64 = {
    "five": ['lle', 'offspring', 'depends', 'covers', 'bunge', 'louth', 'herat', 'beit'],
    "of " : ['regularity', 'carrie', 'disagrees', 'scoring', 'classification', 'hidalgo', 'churchmen', 'kusakabe'],
    "going" : ['mathieu', 'tobit', 'crusade', 'schoolchildren', 'widescreen', 'pasternak', 'dorpat', 'ane'],
    "hardware" : ['dissatisfied', 'clem', 'tatar', 'kites', 'livy', 'honored', 'liver', 'plutarch'],
    "american" : ['ta', 'outflow', 'serbs', 'holloway', 'nonetheless', 'leah', 'maclaurin', 'exchange'],
    "britain" : ['reserving', 'eris', 'machinery', 'nansen', 'surprised', 'glad', 'lizard', 'precisely'] 
}

kraj_model_kont_3_neg_64 = {
    "five" : ['four', 'six', 'three', 'two', 'seven', 'eight', 'one', 'zero'],
    "of"  :  ['the', 'and', 'with', 'in', 'including', 'a', 'from', 'an'],
    "going": ['major', 'both', 'by', 'an', 'no', 'theory', 'made', 'off'],
    "hardware" : ['form', 'or', 'where', 'theory', 'often', 'will', 'because', 'most'],
    "american" : ['john', 'in', 'british', 'german', 's', 'UNK', 'after', 'on'],
    "britain"  : ['now', 'later', 'within', 'up', 'in', 'since', 'people', 'are']    
}
kraj_model_kont_4_neg_64 = {
    'five' : ['four', 'three', 'six', 'two', 'eight', 'one', 'seven', 'nine'],
    'of' : ['and', 'the', 'a', 'for', 'as', 'also', 'to', 'including'],
    'going' : ['his', 'black', 'he', 'after', 'up', 'life', 'made', 'book'],
    'hardware' : ['i', 'computer', 'used', 'a', 'sometimes', 'development', 'usually', 'may'],
    'american' : ['john', 'd', 'british', 'b', 'early', 'film', 'UNK', 's'],
    'britain' : ['state', 'country', 'high', 'main', 'than', 'people', 'over', 'public']
}

kraj_model_kont_5_neg_64 = {
    'five' : ['six', 'four', 'three', 'seven', 'one', 'eight', 'two', 'zero'],
    'of' : ['and', 'the', 'a', 'to', 'which', 'by', 'an', 'on'],
    'going' : ['king', 'then', 'so', 'more', 'still', 'point', 'set', 'people'],
    'hardware' : ['using', 'while', 'under', 'general', 'economic', 'list', 'history', 'include'],
    'american' : ['english', 'john', 'early', 's', 'since', 'b', 'british', 'french'],
    'britain' : ['original', 'each', 'high', 'through', 'include', 'there', 'when', 'on']
}

kraj_model_kont_3_neg_128 = {
    'five' : ['three', 'four', 'six', 'two', 'seven', 'eight', 'one', 'nine'],
    'of' : ['the', 'including', 'and', 'with', 'in', 'its', 'by', 'became'],
    'going' : ['their', 'own', 'who', 'them', 'him', 'or', 'he', 'order'],
    'hardware' : ['up', 'off', 'due', 'see', 'both', 'from', 'system', 'term'],
    'american' : ['british', 'french', 'german', 'b', 'english', 'john', 'd', 'film'],
    'britain' : ['main', 'both', 'by', 'english', 'law', 'new', 'history', 'great']
    
    }

kraj_model_kont_4_neg_128 = {
    'five' : ['three', 'six', 'four', 'two', 'seven', 'eight', 'one', 'nine'],
    'of' : ['the', 'and', 'with', 'a', 'which', 'was', 'while', 'as'],
    'going' : ['power', 'set', 'means', 'an', 'more', 'country', 'of', 'when'],
    'hardware' : ['game', 'computer', 'history', 'series', 'city', 'former', 'during', 'world'],
    'american' : ['john', 'b', 'british', 'german', 'french', 'd', 's', 'since'],
    'britain' : ['end', 'case', 'at', 'before', 'against', 'when', 'he', 'long']
}

kraj_model_kont_5_neg_128 = {
   'five' : ['three', 'four', 'six', 'eight', 'seven', 'two', 'one', 'nine'],
   'of' : ['the', 'and', 'a', 'for', 'with', 'its', 'as', 'an'],
   'going' : ['british', 's', 'when', 'had', 'for', 'international', 'his', 'may'],
   'hardware' : ['church', 'became', 'life', 'other', 'control', 'their', 'several', 'without'],
   'american' : ['john', 'd', 'british', 'early', 'b', 'french', 's', 'one'],
   'britain' : ['for', 'each', 'second', 'international', 'over', 'world', 'general', 'north'] 
}

kraj_model_kont_3_neg_128_imrved = {
    'five' : ['three', 'four', 'six', 'eight', 'seven', 'two', 'nine', 'one'],
    'of' : ['the', 'its', 'including', 'and', 'within', 'most', 'for', 'from'],
    'going' : ['being', 'out', 'this', 'began', 'means', 'however', 'that', 'research'],
    'hardware' : ['way', 'data', 'free', 'greek', 'same', 'more', 'type', 'through'],
    'american' : ['born', 'english', 'john', 'british', 'french', 'film', 'german', 'william'],
    'britain' : ['island', 'house', 'third', 'national', 'first', 'british', 'became', 'country']
    }





#%%

import matplotlib.pyplot as plt

iteracije = list(range(1, 700000, 10000))



plt.figure(figsize=(12, 6))
plt.plot(iteracije, pocetni_model_win_3_neg_64,color = "red", label='M1', linewidth=2)
plt.plot(iteracije, model_konekst_win_4_neg_64 ,color = "green", label='M2', linewidth=2)
plt.plot(iteracije, model_kontekst_win_5_neg_64,color = "blue", label='M3', linewidth=2)
plt.plot(iteracije, model_kontekst_3_neg_128[:70],color = "orange", label='M4', linewidth=2)
plt.plot(iteracije, model_kontekst_4_neg_128[:70] ,color = "black", label='M5', linewidth=2)
plt.plot(iteracije, model_kontekst_5_neg_128[:70],color = "yellow", label='M6', linewidth=2)


plt.xlabel('Iteracije', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.title('Skip-Gram model trening loss kroz iteracije', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)



plt.tight_layout()
plt.show()


#%%
print(len(model_kontekst_3_neg_128[:71]))
print(len(model_kontekst_4_neg_128))
print(len(model_kontekst_5_neg_128))


iteracije = list(range(200000, 700000, 10000))


plt.figure(figsize=(12, 6))
plt.plot(iteracije, pocetni_model_win_3_neg_64[20:70],color = "red", label='M1', linewidth=2)
plt.plot(iteracije, model_konekst_win_4_neg_64[20:70] ,color = "green", label='M2', linewidth=2)
plt.plot(iteracije, model_kontekst_win_5_neg_64[20:70],color = "blue", label='M3', linewidth=2)
plt.plot(iteracije, model_kontekst_3_neg_128[20:70],color = "orange", label='M4', linewidth=2)
plt.plot(iteracije, model_kontekst_4_neg_128[20:70] ,color = "black", label='M5', linewidth=2)
plt.plot(iteracije, model_kontekst_5_neg_128[20:70],color = "yellow", label='M6', linewidth=2)


plt.xlabel('Iteracije', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.title('Skip-Gram model trening loss kroz iteracije', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)


plt.tight_layout()
plt.show()










