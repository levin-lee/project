import math

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import re
import json
import torch

import random
import pandas as pd
from tqdm import tqdm
from simplet5 import SimpleT5
from torch.utils.data import Dataset
from sklearn.metrics import f1_score


# 使用验证集评估模型
with open("./MAMS/MAMS_test.txt", "r") as f:
    file = f.readlines()
    val_data_formatted = []
    for line in file:
        s,t,p = line.strip().split("\001")
        x = s + "The sentiment polarity of " + t.lower() + " is "
        label = p.strip()
        val_data_formatted.append([x, label])

test_df = pd.DataFrame(val_data_formatted, columns=["source_text", "target_text"])

# fetch the path to last model
last_epoch_model = None 
for file in glob("./t5/*"):
    if 'epoch-14' in file:
        last_epoch_model = file
# load the last model
model.load_model("t5", last_epoch_model, use_gpu=True)
# test and save
# for each test data perform prediction
predictions = []
for index, row in test_df.iterrows():
    prediction = model.predict(row['source_text'])[0]
    predictions.append(prediction)
df = test_df.copy()
df['predicted'] = predictions
print("===f1-score====")
print(f1_score(df['target_text'], df['predicted'], average='macro'))
print("===accuracy====")
print(accuracy_score(df['target_text'], df['predicted']))

df.to_csv(f"result_prediction_t5.csv", index=False)
# clean the output
# !rm -rf ./outputs