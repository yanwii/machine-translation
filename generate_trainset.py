import os
import json
trainset_dir = "tmp/"
trainset_io = open(os.path.join(trainset_dir, "translation2019zh_train.json"))
devset_io = open(os.path.join(trainset_dir, "translation2019zh_valid.json"))

data_dir = "tmp/"
train_en_io = open("tmp/raw-train.zh-en.en", "w")
train_zh_io = open("tmp/raw-train.zh-en.zh", "w")

dev_en_io = open("tmp/raw-dev.zh-en.en", "w")
dev_zh_io = open("tmp/raw-dev.zh-en.zh", "w")

def prepare_data(data_io, en_io, zh_io):
    for line in data_io:
        line = line.strip()
        json_obj = json.loads(line)
        en_io.write(json_obj["english"] + "\n")
        zh_io.write(" ".join(list(json_obj["chinese"])) + "\n")
    en_io.close()
    zh_io.close()
    data_io.close()

prepare_data(trainset_io, train_en_io, train_zh_io)
prepare_data(devset_io, dev_en_io, dev_zh_io)
