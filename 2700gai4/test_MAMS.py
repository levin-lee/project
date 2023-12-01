from simpletransformers.seq2seq import Seq2SeqModel
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig,BertTokenizer
from sklearn.metrics import accuracy_score
# logging.basicConfig(level=logging.INFO)
# transformers_logger = logging.getLogger("transformers")
# transformers_logger.setLevel(logging.WARNING)
import torch
import numpy as np
from tqdm.auto import tqdm
label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
def evaluate_val_accuracy(model, val_file_path):
    with open(val_file_path, "r") as f:
        val_data = f.readlines()

    val_data_formatted = []
    for line in val_data:
        s,t,p = line.strip().split("\001")
        x = s + "The sentiment polarity of " + t.lower() + " is "
        label = p
        val_data_formatted.append([x, label])

    val_df = pd.DataFrame(val_data_formatted, columns=["text", "labels"])
    val_df["labels"] = val_df["labels"].map(label_mapping)

    # 使用训练好的模型对验证数据集进行预测
    predictions, logits = model.predict(val_df["text"].tolist())

    # 计算准确率
    accuracy = accuracy_score(val_df["labels"], predictions)
    dataset = val_file_path.strip(".txt").split("_")[-1]
    print(f"{dataset} accuracy:", accuracy)
    return accuracy
def predict_val(model, device,tokenizer,decoder_tok=None):
    candidate_list = ["positive", "neutral", "negative"]

    # model = BartForConditionalGeneration.from_pretrained('./outputs/checkpoint-513-epoch-19')
    model.eval()
    model.config.use_cache = False
    with open("MAMS/MAMS_val.txt", "r") as f:
        fs = list(f.readlines())
    train_data = []
    count = 0
    total = 0
    for line in tqdm(fs):
        total += 1
        score_list2 = []
        line = line.strip()
        x, term, golden_polarity = line.split("\001")[0], line.split("\001")[1], line.split("\001")[2]
        input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids']

        target_list = ["The sentiment polarity of " + term.lower() + " is " + candi.lower() + " ." for candi in
                       candidate_list]
        if decoder_tok:
            output_ids = decoder_tok(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
        else:
            output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
        with torch.no_grad():
            output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]
            logits = output.softmax(dim=-1)
        for i in range(3):
            score = 1
            for j in range(logits[i].shape[0] - 2):
                score *= logits[i][j][output_ids[i][j + 1]]
            score_list2.append(score.cpu().numpy())
        score_list = score_list2

        predict = candidate_list[np.argmax(score_list)]
        if predict == golden_polarity:
            count += 1
    print(f"val {count/total}")
    return count/total

def predict_test(model, device,tokenizer,decoder_tok=None):
    candidate_list = ["positive", "neutral", "negative"]
    model.eval()
    print("start test")
    model.config.use_cache = False
    with open("MAMS/MAMS_test.txt", "r") as f:
        fs = list(f.readlines())
    train_data = []
    count = 0
    total = 0
    for line in tqdm(fs):
        total += 1
        # score_list = []
        score_list1 = []
        score_list2 = []
        score_list3 = []
        score_list4 = []
        score_list5 = []
        line = line.strip()
        x, term, golden_polarity = line.split("\001")[0], line.split("\001")[1], line.split("\001")[2]
        input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids']


        target_list = ["The sentiment polarity of " + term.lower() + " is " + candi.lower() + " ." for candi in
                       candidate_list]
        if decoder_tok:
            output_ids = decoder_tok(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
        else:
            output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
        with torch.no_grad():
            output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]
            logits = output.softmax(dim=-1).to('cpu').numpy()
        for i in range(3):
            score = 1
            for j in range(logits[i].shape[0] - 2):
                score *= logits[i][j][output_ids[i][j + 1]]
            score_list2.append(score)
        score_list = score_list2

        predict = candidate_list[np.argmax(score_list)]
        if predict == golden_polarity:
            count += 1
    print(f"test {count/total}")
    return count/total
