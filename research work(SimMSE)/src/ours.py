import SIF
import feature_select_method as method
import numpy as np
import scipy
import pickle
def load_pkl(filename):
    with open(filename, 'rb') as f:
        t = pickle.load(f)
    return t

def transform_and_method1(name,c,n_compment,manifold_num):
    """
    name: 使用方法的名字
    c：需要使用的批次
    n_compment: 每次降维的大小
    """
    vec_all = load_pkl(r'D:\PyCharmProject\Whitening\BERT-whitening\manifold_learn\data\buffer_data\vec_all.pkl')
    vocab, ivocab, W, words = vec_all['vocab'], vec_all['ivocab'], vec_all['W'], vec_all['words']
    W = np.array(W)
    print(W.shape)
    start = 29820
    w = W[start:start+1000]
    clf = method.method(name, w, n_compment,manifold_num)
    print(type(c), len(c))
    if name == 'isomap':
        x = clf.fit_transform(c)
    elif name == 'lpp':
        x = clf.fit_transform(c)
    else:
        x = np.array(clf.transform(c))
    return x

def transform_and_method(name,c,n_compment,manifold_num):
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

def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation

