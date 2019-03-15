# -*- coding:utf-8 -*-
'''
@Author: yanwii
@Date: 2019-03-14 14:50:26
'''
from argparse import ArgumentParser
import jieba

vocab_size = 50000
vocab = set()
with open("tmp/raw-train.zh-en.zh") as fopen:
    for line in fopen:
        line = line.strip()
        words = list(line)
        for word in words:
            vocab.add(word)
        if len(vocab) > vocab_size:
            break

with open("data/vocab.enzh-sub-zh.{}".format(vocab_size), "w") as fopen:
    fopen.write("<pad>\n<EOS>\nUNK\n" + "\n".join(list(vocab)))