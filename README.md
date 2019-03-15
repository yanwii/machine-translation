# 在tensor2tensor中使用自己的语料实现中英文翻译

[tensor2tensor](github.com/tensorflow/tensor2tensor) 是谷歌提出的一个transformer模型。其结构与end-to-end模型类似，但结构中不再使用RNN作为基础神经元，而是采用self-attention自注意力机制来实现上下文信息的传递  
具体可以参考论文[Attention Is All You Need](https://arxiv.org/abs/1706.03762)  
transfomer的优势在于它不再像RNN那样具有时序行，整个运算都是并行的。

# requirements

- tensor2tensor
- tensorflow
- subword-nmt

安装

    pip3.6 install -r requirements.txt

# 流程

tensor2tensor训练模型分为以下两个步骤  
1.数据准备 t2t-datagen  
2.模型训练 t2t-trainer  

# 数据准备
首先需要准备好语料，这里使用的是[nlp_chinese_corpus](https://github.com/brightmart/nlp_chinese_corpus)中的翻译语料,下载完成解压到项目根目录 **tmp** 下  
使用 **generate_trainset.py** 生成训练数据 **raw-train.zh-en** 和 **raw-dev.zh-en**

    def prepare_data(data_io, en_io, zh_io):
        for line in data_io:
            line = line.strip()
            json_obj = json.loads(line)
            en_io.write(json_obj["english"] + "\n")
            zh_io.write(" ".join(list(json_obj["chinese"])) + "\n")
        en_io.close()
        zh_io.close()
        data_io.close()

由于中文使用的是字符特征，所以保存的时候以空格隔开，方便以后读取,当然也可以使用分词工具，但都需要以空格分开。  

    python3.6 generate_trainset.py

接下来准备 vocab
英文通过 **subword-nmt** 生成词典，然后使用 **generate_en_vocab.py** 整理词典

    subword-nmt get-vocab --input tmp/raw-train.zh-en.en --output en.vocab
    python3.6 generate_en_vocab.py
    python3.6 generate_zh_vocab.py

至此，所有的语料都准备好了，但如果使用自己的数据，就需要注册自己的problem
创建一个 **TranslateEnzhSub50k** 任务, 并在 **user_dir/\_\_init\_\_.py** 中导入

加载用户字典通过以下方法实现  

    def get_vocab(self, data_dir, is_target=False):
        vocab_filename = os.path.join(data_dir, self.target_vocab_name if is_target else self.source_vocab_name)
        if not tf.gfile.Exists(vocab_filename):
            raise ValueError("Vocab %s not found" % vocab_filename)
        return text_encoder.TokenTextEncoder(vocab_filename, replace_oov="UNK")

使用 **t2t-datagen** 生成训练数据
    
    t2t-datagen --data_dir=data/ --problem=translate_enzh_sub50k --t2t_usr_dir=user_dir --tmp_dir=tmp/
    
    参数说明
    --data_dir      生成的训练数据的目录
    --problem       自定义的problem名
    --t2t_usr_dir   problem目录
    --tmp_dir       数据目录

# 训练
    t2t-trainer --data_dir=data   --output_dir=model   --problem=translate_enzh_sub50k   --model=transformer   --hparams_set=transformer_big   --train_steps=200000   --eval_steps=100 --t2t_usr_dir=user_dir --tmp_dir=tmp/ --decode_hparams="batch_size=1024"






