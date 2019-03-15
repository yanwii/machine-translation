# -*- coding:utf-8 -*-
'''
@Author: yanwii
@Date: 2019-03-15 10:51:48
'''

vocab_size = 50000
vocab = set()
with open("tmp/en.vocab") as fopen:
    for line in fopen:
        line = line.strip()
        word = line.split()[0]
        vocab.add(word)
        if len(vocab) > vocab_size:
            break

with open("data/vocab.enzh-sub-en.{}".format(vocab_size), "w") as fopen:
    fopen.write("<pad>\n<EOS>\nUNK\n" + "\n".join(list(vocab)))