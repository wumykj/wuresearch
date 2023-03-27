
import nltk
"""
给定一个总语料库句子矩阵，输入为[sentence1, sentence2,...,sentencen]
输出给定任意句子的字频率信息。
"""
## 初始化句子语料库的参数
## tokens=['i','love',...,'you']
## 初始无监督数据库c,例如以下形式tokens的每个单词为一个字符流的形式。
## data_dict[w]=v 是一个单词w在整个语料库的计数，data_len为整个语料库的长度
tokens = ['i','love','you']
Freq_dist_nltk = nltk.FreqDist(tokens)
data_dict= Freq_dist_nltk
data_len= len(data_dict)

class sent:
    def __init__(self, s):
        self.sentence = s
    def cout_sentence_frequency(self,s):
        s_cout = 0
        for w in s:
            s_cout += data_dict[w]/data_len
        return s_cout

def sentence_sampling(S,n):
    t = []
    for s in S:
        # 计算句频分数，重排序
        a = sent(s)
        s_cout = a.cout_sentence_frequency(s)
        t.append([s,s_cout])
    ans = sorted(t, key=(lambda x: x[1]))
    # 取前n个句子
    res = ans[0:n]
    return res