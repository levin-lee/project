import math

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
# from sklearn.metrics import accuracy_score
# import
import re
import json
import torch

import random
import pandas as pd
from tqdm import tqdm
from simplet5 import SimpleT5
from torch.utils.data import Dataset
# from sklearn.metrics import f1_score
# from sklearn.model_selection import train_test_split

import torch
import json
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# 使用验证集评估模型
# 读取训练数据并转换格式
with open("./MAMS/MAMS_train.txt", "r") as f:
    file = f.readlines()
    train_data = []
    for line in file:
        s,t = line.strip().split("\001")
        x = s + " ".join(t.split()[:-2])
        label = t.split()[-2].strip()
        train_data.append([x, label])

train_df = pd.DataFrame(train_data, columns=["source_text", "target_text"])

## 
with open("./MAMS/MAMS_val.txt", "r") as f:
    file = f.readlines()
    val_data_formatted = []
    for line in file:
        s,t,p = line.strip().split("\001")
        x = s + "The sentiment polarity of " + t.lower() + " is "
        label = p.strip()
        val_data_formatted.append([x, label])

val_df = pd.DataFrame(val_data_formatted, columns=["source_text", "target_text"])

# 将标签转换为整数
# label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
# train_df["target_text"] = train_df["target_text"].map(label_mapping)

# train_data, test_data = train_test_split(train_df,shuffle=True, test_size=0.2, random_state=2023)

# outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels) # 训练
# out = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=2) # 推理
# model = T5ForConditionalGeneration.from_pretrained('t5-base')
# tokenizer = T5Tokenizer.from_pretrained('t5-base')
# device = torch.device('cuda')
# model.to(device)

from glob import glob
# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# 设置模型参数
# for trial_no in range(3):
    # create data 
    # load model
model = SimpleT5()
model.from_pretrained(model_type="t5", model_name="t5-base")
epoch = 15
# train model
model.train(train_df, #=train_data,
            val_df,
            source_max_token_len=300, 
            target_max_token_len=3, 
            batch_size=8,
            max_epochs=epoch, 
            outputdir = "./t5",
            use_gpu=True,
            save_only_last_epoch=True)
