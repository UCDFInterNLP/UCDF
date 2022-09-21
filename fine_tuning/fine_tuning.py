import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, \
                        RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
# from tqdm import tqdm
import torch
from typing import List, Dict, Tuple, Type, Union
from torch.utils.data import Dataset, DataLoader
from numpy import ndarray
from torch import Tensor, device
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datasets import load_dataset
from evaluate import load
from sklearn.metrics import classification_report, precision_recall_curve, auc
import matplotlib.pyplot as plt
from sklearn import metrics


# evaluation metric
f1_metric = load('f1')
acc_metric = load('accuracy')
rec_metric = load('recall')
prec_metric = load('precision')

class contentDataset(Dataset):
    def __init__(self, file, tok, max_len, pad_index=None):
        super().__init__()
        self.tok =tok
        self.max_len = max_len
        self.content = pd.read_csv(file)
        self.len = self.content.shape[0]
        self.pad_index = self.tok.pad_token
    
    def add_padding_data(self, inputs, max_len):
        if len(inputs) < max_len:

            pad = np.array([0] * (max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
            return inputs
        else:
            inputs = inputs[:max_len]
            return inputs
    
    def __getitem__(self,idx):
        instance = self.content.iloc[idx]

        text = instance['text']
        input_ids = self.tok.encode(text)
        
        input_ids = self.add_padding_data(input_ids, max_len=self.max_len)
        label_ids = instance['label']

        return {"encoder_input_ids" : np.array(input_ids, dtype=np.int_),
                "label" : np.array(label_ids,dtype=np.int_)}
        
    def __len__(self):
        return self.len

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=50, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            print('terminating because of early stopping!')
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)

class TestDataset(Dataset):
    def __init__(self, dataset, tok, max_len, pad_index=None):
        super().__init__()
        self.tok =tok
        self.max_len = max_len
        self.content = dataset
        self.len = self.content.shape[0]
        self.pad_index = self.tok.pad_token
    
    def add_padding_data(self, inputs, max_len):
        if len(inputs) < max_len:
            pad = np.array([0] * (max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
            return inputs
        else:
            inputs = inputs[:max_len]
            return inputs
    
    def __getitem__(self,idx):
        instance = self.content.iloc[idx]
        text = instance['text']
        input_ids = self.tok.encode(text)
        
        input_ids = self.add_padding_data(input_ids, max_len=self.max_len)
        return {"encoder_input_ids" : np.array(input_ids, dtype=np.int_)}
                
    def __len__(self):
        return self.len

class HateEncoder(object):
    def __init__(self, ckpt, dataset, existed):
        if existed:
            self.tokenizer = RobertaTokenizer.from_pretrained(ckpt)
            self.model = RobertaForSequenceClassification.from_pretrained(ckpt)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(simcse_ckpt)
            self.model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2)
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dataset = dataset

    def detect(self):

        device = self.device
        test_setup = TestDataset(dataset= self.dataset, tok=self.tokenizer, max_len=128)
        test_dataloader = DataLoader(test_setup, batch_size = 4, shuffle=False)
        pred = []
        logit_socre = []
        self.model.to(device)
        self.model.eval()
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            encoder_attention_mask = batch["encoder_input_ids"].ne(0).float().to(device)
            with torch.no_grad():
                outputs = self.model(batch['encoder_input_ids'], attention_mask=encoder_attention_mask)
            
            logits = outputs.logits
            logit_socre.append(logits.cpu())
            
            predictions = torch.argmax(logits, dim=-1)
            pred.append(predictions)

        logit_socre = torch.cat(logit_socre,0)
        logit_socre = np.array(logit_socre.cpu())

        pred = torch.cat(pred, 0)
        pred = np.array(pred.cpu())

        return pred, logit_socre

def evaluate_result(ckpt,dataset, existed):
    pretrained_hate = HateEncoder(ckpt=ckpt, dataset=dataset, existed=existed)

    # 0: normal, 1: hate
    result, logit_socre = pretrained_hate.detect()

    f1_score = f1_metric.compute(predictions=result, references=dataset['label'])
    acc_score = acc_metric.compute(predictions=result, references=dataset['label'])
    rec_score = rec_metric.compute(predictions=result, references=dataset['label'])
    prec_score = prec_metric.compute(predictions=result, references=dataset['label'])
    
    return result, logit_socre, f1_score, acc_score, rec_score, prec_score

def draw_precision_recall_curve(pred, dataset):
    p,r,_ = precision_recall_curve(pred, dataset['label'])
    plt.plot(p, r, marker='.', label='BERT')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    auc_score = auc(r, p)
    print('auc score: ',auc_score)

def draw_roc_curve(prob, dataset):
    fpr, tpr, _ = metrics.roc_curve(dataset['label'].astype(int),  prob)
    auc = metrics.roc_auc_score(dataset['label'].astype(int),  prob)
    #create ROC curve
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()

def main():

    # load Davidson et al. dataset
    davidson = pd.read_csv('/anaconda3/envs/paper/DPR/fine_tuning/data/davidson_et.csv')
    davidson = davidson[['hate_speech','tweet']]

    davidson_df = pd.DataFrame(index=range(0,davidson.shape[0]), columns=['text','label'])

    for i in range(davidson.shape[0]):
        row = davidson.iloc[i]
        if row['hate_speech'] >= 1:
            davidson_df['label'][i] = 1
            davidson_df['text'][i] = row['tweet']
        else:
            davidson_df['label'][i] = 0
            davidson_df['text'][i] = row['tweet']

    davidson_df = davidson_df.sample(n=int(davidson_df.shape[0]*0.1259), random_state=9786)
    davidson_df = davidson_df.sample(frac=1).reset_index(drop=True)

    # load tweets_hate_speech_detection dataset
    twit = load_dataset("tweets_hate_speech_detection")

    twit = twit['train']
    twit_df = pd.DataFrame(index=range(0,twit.shape[0]), columns=['text','label'])

    for i in range(twit.shape[0]):
        text = twit[i]['tweet']
        label = twit[i]['label']
        if int(label) == 0 or 1:
            twit_df['text'][i] = text
            twit_df['label'][i] = label
        else:
            continue


    twit_df = twit_df.sample(n=int(twit_df.shape[0]*0.1), random_state=1234)
    twit_df = twit_df.sample(frac=1).reset_index(drop=True)


    # build customized dataset

    # cosssim - min
    global simcse_ckpt 
    simcse_ckpt = 'princeton-nlp/sup-simcse-bert-base-uncased'

    # Appendix Religion 
    passages = pd.read_csv('/anaconda3/envs/paper/DPR/retrieved_text_output/simcse/Appendix/religion_cos_sim_avg_simcse_positive.csv')

    #cos-sim / avg
    # passages = pd.read_csv('/anaconda3/envs/paper/DPR/retrieved_text_output/simcse/cos_sim/cos_sim_avg_simcse_positive.csv')

    #cos-sim / min
    # passages = pd.read_csv('/anaconda3/envs/paper/DPR/retrieved_text_output/simcse/cos_sim/cos_sim_min_simcse_positive.csv')

    #dot-product / avg
    # passages = pd.read_csv('/anaconda3/envs/paper/DPR/retrieved_text_output/simcse/dot_product/dot_product_avg_simcse_positive.csv')

    #dot-product / min
    # passages = pd.read_csv('/anaconda3/envs/paper/DPR/retrieved_text_output/simcse/dot_product/dot_product_min_simcse_positive.csv')

    passages.drop(['Unnamed: 0'],axis=1, inplace=True)

    # hate speech detection
    # query = ['a direct attack against people — rather than concepts or institutions— based on protected characteristics: race, ethnicity, national origin, disability, religious affiliation, caste, sexual orientation, sex, gender identity, and severe disease. ',
    #         "content promoting violence or hatred against individuals or groups based on any of the following attributes: race or ethnic origin, religion, disability, gender, age, veteran status, or sexual orientation/gender identity.",
    #         "violence against or directly attack or threaten other people based on race, ethnicity, national origin, caste, sexual orientation, gender, gender identity, religious affiliation, age, disability, or serious disease. ",
    #         "hateful and discriminatory speech. any expression that is directed to an individual or group of individuals based upon the personal characteristics of that individual or group. ",
    #         'products that promote, incite, or glorify hatred, violence, racial, sexual, or religious intolerance or promote organizations with such views.',
    #         "Hate speech or content that demeans, defames, or promotes discrimination or violence on the basis of race, color, caste, ethnicity, national origin, religion, sexual orientation, gender identity, disability, or veteran status, immigration status, socio-economic status, age, weight or pregnancy status",
    #         "any activity, post any User Content, or register or use a username, which is or includes material that is offensive, abusive, defamatory, pornographic, threatening, or obscene, or advocates or incites violence."]

    # Relgiion 
    query = ['Judaism is the world’s oldest monotheistic religion, dating back nearly 4,000 years. ',
            'Christianity is the most widely practiced religion in the world, with more than 2 billion followers. ',
            'Islam is the second largest religion in the world after Christianity, with about 1.8 billion Muslims worldwide.',
            'Buddhism is a faith that was founded by Siddhartha Gautama (“the Buddha”) more than 2,500 years ago in India. ',
            'Hinduism is the world’s oldest religion, according to many scholars, with roots and customs dating back more than 4,000 years. '
                        ]
                        
    passages = passages.append(pd.DataFrame(query,columns=['text']))

    total_passages = list(passages.text)
    for q in query:
        total_passages.append(q)

    # Appendix Religion 
    negative = pd.read_csv('/anaconda3/envs/paper/DPR/retrieved_text_output/simcse/Appendix/religion_cos_sim_avg_simcse_negative.csv')

    #cos-sim / avg
    # negative = pd.read_csv('/anaconda3/envs/paper/DPR/retrieved_text_output/simcse/cos_sim/cos_sim_avg_simcse_negative.csv')

    #cos-sim / min
    # negative = pd.read_csv('/anaconda3/envs/paper/DPR/retrieved_text_output/simcse/cos_sim/cos_sim_min_simcse_negative.csv')

    #dot-product / avg
    # negative = pd.read_csv('/anaconda3/envs/paper/DPR/retrieved_text_output/simcse/dot_product/dot_product_avg_simcse_positive.csv')

    #dot-product / min
    # negative = pd.read_csv('/anaconda3/envs/paper/DPR/retrieved_text_output/simcse/dot_product/dot_product_min_simcse_positive.csv')

    negative.drop(['Unnamed: 0'],axis=1, inplace=True)


    positive_df = passages.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
    negative_df = negative.sample(n=positive_df.shape[0], random_state=1234)



    def build_dataset(pos_df, neg_df):
        # --> in this example
        # pos: hate text 
        # neg: normal text
        pos_df['label'] = 1
        neg_df['label'] = 0
        data = pd.concat([pos_df, neg_df],axis=0)
        data = data.sample(frac=1).reset_index(drop=True)
        return data

    dataset = build_dataset(positive_df, negative_df)
    dataset = dataset.dropna(axis=0)



    # fine-tuning
    model = AutoModelForSequenceClassification.from_pretrained(simcse_ckpt, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(simcse_ckpt)


    train_num = int(len(dataset)*0.8)
    trainset = dataset.iloc[:train_num]
    validset = dataset.iloc[train_num:]

    trainset.to_csv('simcse_enc_trainset.csv')
    validset.to_csv('simcse_enc_validset.csv')

    train_setup = contentDataset(file = "simcse_enc_trainset.csv",tok = tokenizer, max_len = 128)
    valid_setup = contentDataset(file = "simcse_enc_validset.csv",tok = tokenizer, max_len = 128)

    train_dataloader = DataLoader(train_setup, batch_size=32, shuffle=True)
    valid_dataloader = DataLoader(valid_setup, batch_size=32, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=5e-5)


    num_epochs = 5
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)


    progress_bar = tqdm(range(num_training_steps))

    def train_one_epoch(epoch_index):
        running_loss = 0.
        last_loss = 0.

        for i, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            encoder_attention_mask = batch["encoder_input_ids"].ne(0).float().to(device)
            optimizer.zero_grad()
            
            outputs = model(batch['encoder_input_ids'], attention_mask=encoder_attention_mask, labels=batch['label'])

            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            
            progress_bar.update(1)

            running_loss += loss.item()
            if i % 10 == 9:
                last_loss = running_loss / 9
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0.

        return last_loss

    progress_bar = tqdm(range(num_training_steps))
    loss_list = []
    best_vloss = 99999999
    vloss_list = []
    es = EarlyStopping(patience=2)

    model.train()
    for epoch in range(num_epochs):
        print('EPOCH {}:'.format(epoch + 1))

        model.train(True)
        avg_loss = train_one_epoch(epoch)
        loss_list.append(avg_loss)

        model.train(False)
        running_vloss = 0.0
        for i, batch in enumerate(valid_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            encoder_attention_mask = batch["encoder_input_ids"].ne(0).float().to(device)
            with torch.no_grad():
                voutputs = model(batch['encoder_input_ids'], attention_mask=encoder_attention_mask, labels=batch['label'])
            vloss = voutputs.loss
            running_vloss += vloss


        avg_vloss = running_vloss / (i + 1)
        vloss_list.append(avg_vloss)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        if avg_vloss < best_vloss:
            
            best_vloss = avg_vloss
            print('best updated!')
            print('bets vloss: ',best_vloss)
            model.save_pretrained(f"/anaconda3/envs/paper/DPR/fine_tuning/ckpt/simcse_cos_avg_religion")

        if es.step(best_vloss.item()):
            print('EARLY STOPPING!')
            break


    model = AutoModelForSequenceClassification.from_pretrained("/anaconda3/envs/paper/DPR/fine_tuning/ckpt/simcse_cos_avg_religion", num_labels=2)
    model.to(device)
    pred = []
    ref = []

    model.eval()
    for batch in valid_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        encoder_attention_mask = batch["encoder_input_ids"].ne(0).float().to(device)
        with torch.no_grad():
            outputs = model(batch['encoder_input_ids'], attention_mask=encoder_attention_mask, labels=batch['label'])
            
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        pred.append(predictions)
        ref.append(batch['label'])

    pred = torch.cat(pred, 0)
    ref = torch.cat(ref, 0)

    f1_score = f1_metric.compute(predictions=pred, references=ref)
    acc_score = acc_metric.compute(predictions=pred, references=ref)
    rec_score = rec_metric.compute(predictions=pred, references=ref)
    prec_score = prec_metric.compute(predictions=pred, references=ref)

    print('f1: ',f1_score)
    print('acc: ',acc_score)
    print('recall: ',rec_score)
    print('precision: ',prec_score)


    # # Twitter Dataset Result
    # # existed method / trained with jigsaw dataset
    # # SkolkovoInstitute/roberta_toxicity_classifier
    # # tomh/toxigen_roberta

    # pred, logit, f1_score, acc_score, rec_score, prec_score = evaluate_result(ckpt='mrm8488/distilroberta-finetuned-tweets-hate-speech', dataset=twit_df, existed=True)

    # prob = torch.nn.functional.softmax(torch.tensor(logit), dim=1)
    # prob = prob[::,1]

    # print(f1_score, acc_score, rec_score, prec_score)

    # draw_precision_recall_curve(pred, twit_df)
    # print('='*30)
    # draw_roc_curve(np.array(prob), twit_df)

    # # fine-tuned results

    # pred, logit, f1_score, acc_score, rec_score, prec_score = evaluate_result(ckpt='/anaconda3/envs/paper/DPR/discussed_0905/ckpt/simcse_heuristic_250', dataset=twit_df, existed=False)

    # prob = torch.nn.functional.softmax(torch.tensor(logit), dim=1)
    # prob = prob[::,1]

    # print(f1_score, acc_score, rec_score, prec_score)

    # draw_precision_recall_curve(pred, twit_df)
    # print('='*30)
    # draw_roc_curve(np.array(prob), twit_df)


    # # Davidson Dataset Result
    # # existed method / trained with jigsaw dataset

    # pred, logit, f1_score, acc_score, rec_score, prec_score = evaluate_result(ckpt='SkolkovoInstitute/roberta_toxicity_classifier', dataset=davidson_df, existed=True)

    # prob = torch.nn.functional.softmax(torch.tensor(logit), dim=1)
    # prob = prob[::,1]

    # print(f1_score, acc_score, rec_score, prec_score)

    # draw_precision_recall_curve(pred, davidson_df)
    # print('='*30)
    # draw_roc_curve(np.array(prob), davidson_df)


    # Appendix - religion example
    appendix_df = pd.read_csv('appendix_religioni_custom_data.csv')

    # fine-tuned time 13 min

    pred, logit, f1_score, acc_score, rec_score, prec_score = evaluate_result(ckpt='/anaconda3/envs/paper/DPR/discussed_0905/ckpt/simcse_cos_avg_religion', dataset=appendix_df, existed=False)

    prob = torch.nn.functional.softmax(torch.tensor(logit), dim=1)
    prob = prob[::,1]

    print(f1_score, acc_score, rec_score, prec_score)

    print('='*30)
    draw_roc_curve(np.array(prob), appendix_df)


if __name__ == "__main__":
    main()

