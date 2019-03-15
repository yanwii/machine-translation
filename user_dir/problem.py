from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import os
import tarfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.data_generators import tokenizer
from tensor2tensor.utils import registry
import tensorflow as tf
 
from collections import defaultdict
 

ENZH_RAW_DATASETS = {
    "TRAIN": "raw-train.zh-en",
    "DEV": "raw-dev.zh-en"
}
 
 
def get_enzh_raw_dataset(directory, filename):
    train_path = os.path.join(directory, filename)
    if not (tf.gfile.Exists(train_path + ".en") and
            tf.gfile.Exists(train_path + ".zh")):
        raise Exception("there should be some training/dev data in the tmp dir.")
    return train_path
  
 
@registry.register_problem
class TranslateEnzhSub50k(translate.TranslateProblem):
 
    @property
    def approx_vocab_size(self):
        return 50000
 
    @property
    def source_vocab_name(self):
        return "vocab.enzh-sub-en.%d" % self.approx_vocab_size
 
    @property
    def target_vocab_name(self):
        return "vocab.enzh-sub-zh.%d" % self.approx_vocab_size
 
    def get_vocab(self, data_dir, is_target=False):
        vocab_filename = os.path.join(data_dir, self.target_vocab_name if is_target else self.source_vocab_name)
        if not tf.gfile.Exists(vocab_filename):
            raise ValueError("Vocab %s not found" % vocab_filename)
        return text_encoder.TokenTextEncoder(vocab_filename, replace_oov="UNK")
 
    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        train = dataset_split == problem.DatasetSplit.TRAIN
        dataset_path = (ENZH_RAW_DATASETS["TRAIN"] if train else ENZH_RAW_DATASETS["DEV"])
        train_path = get_enzh_raw_dataset(tmp_dir, dataset_path)
        return text_problems.text2text_txt_iterator(train_path + ".en",
                                                    train_path + ".zh")
 
    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        encoder = self.get_vocab(data_dir)
        target_encoder = self.get_vocab(data_dir, is_target=True)
        return text_problems.text2text_generate_encoded(generator, encoder, target_encoder,
                                                        has_inputs=self.has_inputs)
 
    def feature_encoders(self, data_dir):
        source_token = self.get_vocab(data_dir)
        target_token = self.get_vocab(data_dir, is_target=True)
        return {
            "inputs": source_token,
            "targets": target_token,
        }
