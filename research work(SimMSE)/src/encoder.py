import numpy as np
import numpy
import os, sys
import pickle
import feature_select_method as method
import pylab as pl
import SIF
import torch
# import DataPreprocessing

def convert_to_vecs2(tokenizer,model,flag,pooling,data):
    """
    输入：
        data:为一行文本1，文本2，相似度
        tokenizer:分词器
        model：模型
        flag：根据模型来选择对应的池化方案
        pooling：对应的池化方案
    输出：
        vecs:vecs【i】表示第i个样本，vecs【i】【0】表示a_vec，vecs【i】【1】表示b_vec
        labels：样本对应的标签
    """
    vecs, labels = [], []
    for d in data:
        a_vec = select_pooling(tokenizer, model, pooling, d[0], flag)
        b_vec = select_pooling(tokenizer, model, pooling, d[1], flag)
        vec = []
        vec.append(a_vec)
        vec.append(b_vec)
        vecs.append(np.array(vec))
        labels.append(d[2])
    return np.array(vecs), np.array(labels)

def convert_to_vecs1(tokenizer,model,flag,pooling,data):
    """
    data:为两个文本加一个相似度
    tokenizer:分词器
    model：模型
    flag：根据模型来选择对应的池化方案
    pooling：对应的池化方案
    """
    a_vecs, b_vecs, labels = [], [], []
    for d in data:
        a_vec = select_pooling(tokenizer, model, pooling, d[0], flag)
        a_vecs.append(a_vec)
        b_vec = select_pooling(tokenizer, model, pooling, d[1], flag)
        b_vecs.append(b_vec)
        labels.append(d[2])
    return np.array(a_vecs), np.array(b_vecs), np.array(labels)

def convert_to_vecs3(tokenizer,model,flag,pooling,data):
    """
    data:为文本数组
    tokenizer:分词器
    model：模型
    flag：根据模型来选择对应的池化方案
    pooling：对应的池化方案
    """
    a_vecs = []
    i = 1
    print("encoding start")
    for d in data:
        a_vec = select_pooling(tokenizer, model, pooling, d, flag)
        a_vecs.append(np.array(a_vec))
    print('encoding end')
    return np.array(a_vecs)

def select_model(method):
    """
    输入：
        method：提供可选的模型名字['bert', 'bert1', 'roberta', 'xlnet', 'unsup_simcse',
                     'sup_simcse', 'bart', 'T5', 'gpt-2']
    输出：
        tokenizer:分词器
        model：模型
        flag：根据模型来选择对应的池化方案
    """
    flag = 1
    if method == 'unsup_simcse':
        from simcse import SimCSE
        flag = 0
        tokenizer = 0
        model = SimCSE("princeton-nlp/unsup-simcse-bert-base-uncased")
    elif method == 'sup_simcse':
        from simcse import SimCSE
        flag = 0
        tokenizer = 0
        model = SimCSE("princeton-nlp/sup-simcse-roberta-base")
    elif method == 'sbert':
        from sentence_transformers import SentenceTransformer
        flag = 0
        tokenizer = 0
        model = SentenceTransformer('all-MiniLM-L6-v2')
    elif method == 'bert':
        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased",output_hidden_states=True)
    elif method == 'roberta':
        from transformers import RobertaTokenizer, RobertaModel
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base',output_hidden_states=True)
    elif method=='xlnet':
        flag = 2
        from transformers import XLNetTokenizer, XLNetModel
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        model = XLNetModel.from_pretrained('xlnet-base-cased',output_hidden_states=True)
    elif method=='bart':
        from transformers import BartTokenizer, BartModel
        flag = 3
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        model = BartModel.from_pretrained("facebook/bart-base",output_hidden_states=True)
    elif method == 'gpt-2':
        from transformers import GPT2Tokenizer, GPT2Model
        flag = 2
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2Model.from_pretrained("gpt2",output_hidden_states=True)
    elif method == 'T5':
        flag = 4
        from transformers import T5Tokenizer, T5EncoderModel
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        model = T5EncoderModel.from_pretrained("t5-base",output_hidden_states=True)
    return tokenizer,model,flag


def select_pooling(tokenizer,model,pooling,d,flag):
    if flag == 0:
        return model_pooling0(tokenizer,model,pooling,d)
    elif flag == 1:
        return model_pooling1(tokenizer,model,pooling,d)
    elif flag == 2:
        return model_pooling2(tokenizer,model,pooling,d)
    elif flag == 3:
        return model_pooling3(tokenizer,model,pooling,d)
    elif flag == 4:
        return model_pooling4(tokenizer,model,pooling,d)

def model_pooling0(tokenizer,model,pooling,d):
    """
       ['unsup_simcse','sup_simcse','sbert']对应的池化方案
    """
    res = model.encode(d).numpy()
    return res

def model_pooling1(tokenizer,model,pooling,d):
    """
           ['bert','roberta']对应的池化方案
    """
    if pooling == 'prompt':
        t = d
        d = "This sentence : " + '"' + t + '"' + " means [MASK]."
    inputs0 = tokenizer(d, return_tensors="pt",padding=True)
    outputs0 = model(**inputs0)
    if pooling == 'cls':
        hidden_states = outputs0.pooler_output
        vec = hidden_states[0]
        res = vec.detach().numpy()
    elif pooling == 'mean_f_l':
        hidden_states = torch.cat((outputs0.hidden_states[0][0], outputs0.hidden_states[-1][0]), 0)
        vec = hidden_states.mean(0)
        res = vec.detach().numpy()
    elif pooling == 'last':
        hidden_states = outputs0.last_hidden_state
        vec = hidden_states[0].mean(0)
        res = vec.detach().numpy()
    elif pooling == 'mean_f1_f2':
        hidden_states = torch.cat((outputs0.hidden_states[0][0], outputs0.hidden_states[1][0]), 0)
        vec = hidden_states.mean(0)
        res = vec.detach().numpy()
    elif pooling == "prompt":
        v1 = outputs0.hidden_states[0][0][-3].unsqueeze(0)
        v2 = outputs0.hidden_states[-1][0][-3].unsqueeze(0)
        hidden_states = torch.cat((v1, v2), 0)
        vec = hidden_states.mean(0)
        res = vec.detach().numpy()
    elif pooling == "all_hidden":
        hidden_states = outputs0.hidden_states
        # 对每一层的向量取平均
        h = [i[0].mean(0) for i in hidden_states]
        res = torch.stack(h)
    return res

def model_pooling2(tokenizer,model,pooling,d):
    """
    'xlnet','gpt-2' 对应的池化方案
    """
    inputs0 = tokenizer(d, return_tensors="pt")
    print(inputs0)
    if inputs0.input_ids.shape[0]>0:
        outputs0 = model(**inputs0)
    else:
        inputs0 = tokenizer('i love you', return_tensors="pt")
        print('encoding error')
        outputs0 = model(**inputs0)
    if pooling == 'cls':
        hidden_states = outputs0.hidden_states
        # 对每一层的所有单词向量取平均
        h = [i[0].mean(0) for i in hidden_states]
        stack1 = torch.stack(h)
        # 对每个层取平均
        vec = stack1.mean(0)
        res = vec.detach().numpy()
    elif pooling == 'mean_f_l':
        hidden_states = torch.cat((outputs0.hidden_states[0][0], outputs0.hidden_states[-1][0]), 0)
        vec = hidden_states.mean(0)
        res = vec.detach().numpy()
    elif pooling == 'last':
        hidden_states = outputs0.last_hidden_state
        vec = hidden_states[0].mean(0)
        res = vec.detach().numpy()
    return res

def model_pooling3(tokenizer,model,pooling,d):
    """
    bart 对应的池化方案
    """
    inputs0 = tokenizer(d, return_tensors="pt")
    outputs0 = model(**inputs0)
    if pooling == 'cls':
        hidden_states = outputs0.encoder_hidden_states
        h = [i[0].mean(0) for i in hidden_states]
        stack1 = torch.stack(h)
        # 对每个层取平均
        vec = stack1.mean(0)
        res = vec.detach().numpy()
    elif pooling == 'mean_f_l':
        hidden_states = outputs0.decoder_hidden_states
        h = [i[0].mean(0) for i in hidden_states]
        stack1 = torch.stack(h)
        # 对每个层取平均
        vec = stack1.mean(0)
        res = vec.detach().numpy()
    elif pooling == 'last':
        hidden_states = outputs0.last_hidden_state
        vec = hidden_states[0].mean(0)
        res = vec.detach().numpy()
    return res

def model_pooling4(tokenizer,model,pooling,d):
    """
    T5 对应的池化方案
    """
    inputs0 = tokenizer(d, return_tensors="pt")
    outputs0 = model(**inputs0)
    if pooling == 'cls':
        hidden_states = outputs0.hidden_states
        h = [i[0].mean(0) for i in hidden_states]
        stack1 = torch.stack(h)
        # 对每个层取平均
        vec = stack1.mean(0)
        res = vec.detach().numpy()
    elif pooling == 'mean_f_l':
        hidden_states = torch.cat((outputs0.hidden_states[0][0], outputs0.hidden_states[-1][0]), 0)
        vec = hidden_states.mean(0)
        res = vec.detach().numpy()
    elif pooling == 'last':
        hidden_states = outputs0.last_hidden_state
        vec = hidden_states[0].mean(0)
        res = vec.detach().numpy()
    return res

if __name__ == '__main__':
    method = 'bert'
    methods = {'unsup_simcse','sup_simcse','sbert','bert','roberta','xlnet','bart','gpt-2','T5'}
    # label = 1
    # task_path = '../data/downstream/MR/'
    # pos, neg = DataPreprocessing.load_sent_data(task_path)
    # t = np.array(pos)
    # print(len(pos))
    # print(method)
    # vecs, label = convert_to_vecs(method,t[0:10],label)
    # print(vecs.shape,label.shape)
    print(type(ord('a')-ord('b')))