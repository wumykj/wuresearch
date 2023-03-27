#! -*- coding: utf-8 -*-
# 简单的线性变换（白化）操作，就可以达到甚至超过BERT-flow的效果。

from utils import *
import os, sys
from simcse import SimCSE
import pickle
import feature_select_method as method
import pylab as pl
import SIF
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import BartTokenizer, BartModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import XLNetTokenizer, XLNetModel
from transformers import GPT2Tokenizer, GPT2Model
from transformers import GPT2TokenizerFast
import torch
from transformers import RobertaTokenizer, RobertaModel
from transformers import XLNetTokenizer, XLNetModel
sys.path.append('..')
from senteval.src import encoder
model_type, task_name, n_components, normalized_by = 'base','STS-12',600,'nli'
n_components = int(n_components)
if n_components < 0:
    if model_type.startswith('base'):
        n_components = 768
    elif model_type.startswith('large'):
        n_components = 1024
# 加载数据集
def obtain_datasets(task_name):
    data_path = '../root/glue/'

    if task_name == 'STS-B':
        datasets = {
            'sts-b-train': load_sts_b_train_data(data_path + 'STS-B/train.tsv'),
            'sts-b-dev': load_sts_b_train_data(data_path + 'STS-B/dev.tsv'),
            'sts-b-test': load_sts_b_test_data(data_path + 'STS-B/sts-test.csv')
        }
    elif task_name.startswith('STS-1'):
        names = sts_12_16_names[task_name]
        datasets = {
            n: load_sts_12_16_data(data_path + task_name + '/STS.input.%s.txt' % n)
            for n in names
        }
    elif task_name == 'SICK-R':
        datasets = {
            'sick-r-train': load_sick_r_data(data_path + 'SICK-R/SICK_train.txt'),
            'sick-r-dev': load_sick_r_data(data_path + 'SICK-R/SICK_trial.txt'),
            'sick-r-test':
                load_sick_r_data(data_path + 'SICK-R/SICK_test_annotated.txt'),
        }
    return datasets


def save_pkl(filename,t):
    with open(filename,'wb') as f:
        pickle.dump(t, f)

def load_pkl(filename):
    with open(filename, 'rb') as f:
        t = pickle.load(f)
    return t



def get_cos_similar_matrix(v1, v2):
    """
    求两个矩阵的余弦相似度，返回一个相似度矩阵
    """
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res

def guiyihua(x):
    """
    对一维度的数据进行归一化处理
    """
    x = np.array(x)
    maxcols = x.max(axis=0)
    mincols = x.min(axis=0)
    x1 = [x[i] / (maxcols - mincols) for i in range(len(x))]
    return np.array(x1)

def transform_and_method(name,c,n_compment):
    """
    使用不同的方法对其进行变换
    """
    if name == 'sif':
        x = SIF.SIF_embedding(c, rmpc=1)
        print("sif is very good")
    else:
        clf = method.method(name,c,n_compment)
        print(type(c),len(c))
        if name=='isomap':
            x = clf.fit_transform(c)
        else:
            x = np.array(clf.transform(c))
    return x

def main():
    ans = {}
    poolings = {'mean_f_l', 'cls', 'last'}
    # poolings = ['cls']
    task_names = ['STS-B', 'STS-12', 'STS-13', 'STS-14', 'STS-15', 'STS-16', 'SICK-R']
    encoder_names = ['bert', 'bert1', 'roberta', 'xlnet', 'unsup_simcse',
                     'sup_simcse', 'bart', 'T5', 'gpt-2']
    encoder_names = ['bert',  'roberta', 'xlnet',  'bart', 'T5', 'gpt-2']
    encoder_names = ['T5']
    f_extension = '.pkl'
    de = 512
    for encoder_name in encoder_names:
        tokenizer, model, flag = encoder.select_model(encoder_name)
        for pooling in poolings:
            for task_name in task_names:
                datasets = obtain_datasets(task_name)
                # 语料向量化
                all_names, all_weights, all_vecs, all_labels = [], [], [], []
                for name, data in datasets.items():
                    a_vecs, b_vecs, labels = encoder.convert_to_vecs1(tokenizer, model, flag, pooling, data)
                    all_names.append(name)
                    all_weights.append(len(data))
                    all_vecs.append((a_vecs, b_vecs))
                    all_labels.append(labels)
                save_pkl('buffer_data/all_vecs'+'_'+encoder_name+'_'+pooling+'_'+task_name+f_extension, all_vecs)
                save_pkl('buffer_data/all_names'+'_'+encoder_name+'_'+pooling+'_'+task_name+f_extension, all_names)
                save_pkl('buffer_data/all_weights'+'_'+encoder_name+'_'+pooling+'_'+task_name+f_extension, all_weights)
                save_pkl('buffer_data/all_labels'+'_'+encoder_name+'_'+pooling+'_'+task_name+f_extension, all_labels)

                all_corrcoefs = []
                all_corrcoefs1 = []
                for (a_vecs, b_vecs), labels in zip(all_vecs, all_labels):
                    a_vecs1 = transform_and_method('sif', a_vecs, de)
                    b_vecs1 = transform_and_method('sif', b_vecs, de)
                    sim0 = get_cos_similar_matrix(a_vecs, b_vecs)
                    sim1 = get_cos_similar_matrix(a_vecs1, b_vecs1)
                    sims0 = np.array([sim0[i][i] for i in range(sim0.shape[0])])  # cos
                    sims1 = np.array([sim1[i][i] for i in range(sim1.shape[0])])
                    labels1 = guiyihua(labels)
                    corrcoef = compute_corrcoef(labels1, sims0)
                    corrcoef1 = compute_corrcoef(labels1, sims1)
                    all_corrcoefs.append(corrcoef)
                    all_corrcoefs1.append(corrcoef1)
                ans[task_name + '_' + encoder_name + '_' + pooling + '_' + 'origin'] = all_corrcoefs
                ans[task_name + '_' + encoder_name + '_' + pooling + '_' + 'sif'] = all_corrcoefs1
                for name, corrcoef in zip(all_names , all_corrcoefs):
                    print('%s: %s' % (name, corrcoef))
    print(ans)
    save_pkl('DE_T5'+f_extension,ans)

if __name__ == '__main__':
    main()