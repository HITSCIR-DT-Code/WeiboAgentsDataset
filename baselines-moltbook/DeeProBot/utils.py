import math
import nltk
import pandas as pd


def entropy(string):
    """字符级香农熵（归一化到字符串长度）"""
    if not string:
        return 0.0
    length = len(string)
    freq = {}
    for ch in string:
        freq[ch] = freq.get(ch, 0) + 1
    h = 0.0
    for count in freq.values():
        p = count / length
        h -= p * math.log(p, 2)
    return h / length


def bigrams_freq(string):
    """bigram 频数之和除以 unique bigram 数量"""
    if not string or len(string) < 2:
        return 0.0
    bigrams_unique = set(nltk.bigrams(string))
    K = len(bigrams_unique)
    if K == 0:
        return 0.0
    freq = dict(pd.DataFrame(list(nltk.bigrams(string))).value_counts())
    C = sum(freq[bg] for bg in bigrams_unique)
    return C / K
