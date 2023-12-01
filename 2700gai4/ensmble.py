from bert import ClassificationModel
from seq2seq_model_M import Seq2SeqModel
from tqdm.auto import tqdm, trange
import torch
import numpy as np
from test_MAMS import label_mapping
import pandas as pd
def accuracy_score(pred,label):
    count = 0
    print(len(pred))
    for idx in range(len(pred)):
        if pred[idx] == label[idx]:
            count+=1
    return count/len(pred)

def classify_predict_logits(model_simple, fs):
    val_data_formatted = []
    for line in fs:
        s,t,p = line.strip().split("\001")
        x = s + "The sentiment polarity of " + t.lower() + " is "
        label = p
        val_data_formatted.append([x, label])

    val_df = pd.DataFrame(val_data_formatted, columns=["text", "labels"])
    val_df["labels"] = val_df["labels"].map(label_mapping)

    # 使用训练好的模型对验证数据集进行预测
    predictions, logits = model_simple.predict(val_df["text"].tolist())
    logits_tensor = torch.tensor(logits,dtype=torch.float32).softmax(dim=-1)
    return logits_tensor.cpu().numpy()

def classify_predict_t5(fp):
    # 使用验证集评估模型
    with open(fp, "r") as f:
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
        if prediction=="negative":
            predictions.append([1,0,0])
        elif prediction=="negative":
            predictions.append([0,1,0])
        else:
            predictions.append([0,0,1])
    return np.array(predictions)

def generate_predict_logits(model_cls,fs):
    model = model_cls.model.cuda()
    tokenizer = model_cls.encoder_tokenizer
    candidate_list = ["negative", "neutral", "positive"]
    count = 0
    total = 0
    device = "cuda"
    res = np.zeros((len(fs),3))
    for line in tqdm(fs):
        total += 1
        score_list2 = []
        line = line.strip()
        x, term, golden_polarity = line.split("\001")[0], line.split("\001")[1], line.split("\001")[2]
        input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids']

        target_list = ["The sentiment polarity of " + term.lower() + " is " + candi.lower() + " ." for candi in
                       candidate_list]
        output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
        with torch.no_grad():
            output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]
            logits = output.softmax(dim=-1)
        for i in range(3):
            score = 1
            for j in range(logits[i].shape[0] - 2):
                score *= logits[i][j][output_ids[i][j + 1]]
            score_list2.append(score.cpu().numpy())
        res[total-1] = np.array(score_list2,dtype=np.float32)

    return res

def test_ensemble(bert_path1,bart_path2,bart_path3=None,weights=[1,1,1]):
    model_args = {
    "max_seq_length": 50,
    "train_batch_size": 10,
    "manual_seed": 42,
    }
    model1 = ClassificationModel("roberta", bert_path1, num_labels=3, args=model_args)
    model2 = Seq2SeqModel(encoder_decoder_type="bart",encoder_decoder_name=bart_path2,args=model_args,)
    model3 = Seq2SeqModel(encoder_decoder_type="marian",encoder_decoder_name=bart_path3,args=model_args,)
    # 将生成模型的预测结果转换为分类任务所需的格式
    def test(fp):
        with open(fp) as f:
            fs_val = list(f.readlines())
        logits1 = classify_predict_logits(model1,fs_val)
        logits2 = generate_predict_logits(model2,fs_val)
        # logits3 = generate_predict_logits(model3,fs_val)
        logits3 = generate_predict_t5(fp)
        
        logits = [logits1,logits2,logits3]
        logits_final = sum([weights[i]*logits[i] for i in range(3)])/(sum(weights))
        label = []
        for line in fs_val:
            golden_polarity = line.split("\001")[2].strip()
            label.append(label_mapping[golden_polarity])
        label = np.array(label,dtype=np.int)
        pred = logits_final.argmax(axis=-1)
        print(accuracy_score(pred,label))
        print("=========end===========")
        file_names = "R%.2d_B%.2d_output.txt" %(10*weights[0],10*weights[1])
        with open(file_names,"w") as f:
            # f.write("=================test valuation===================== /n")
            f.writelines(str(accuracy_score(pred,label)))
        f.close()
    # test("./MAMS/MAMS_val.txt")
    test("./MAMS/MAMS_test.txt")
epoch = 1
for R in np.arange(0,1.1,0.1):
    for B in np.arange(0,1.1,0.1):
        t5_weight = 1 - R -B
        if t5_weight>=0:
            print("==========start {} ============".format(epoch))
            print("Roberta weight is {}, Bart weight is {}".format(R,B))
            
            weights = [R,B,t5_weight]
            test_ensemble("./bert","./bart","./t5",weights)
            epoch += 1
