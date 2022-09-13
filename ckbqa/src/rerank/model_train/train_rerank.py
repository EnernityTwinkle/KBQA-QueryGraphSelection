from __future__ import absolute_import, division, print_function

import yaml
import logging
import os
import random
import sys
import numpy as np
import torch
import math
import pickle
import shutil
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from argparse import ArgumentParser
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)
from src.utils.cal_f1 import cal_f1, cal_f1_with_scores
from src.rerank.model_train.BertEncoderX import BertFor2PairSequenceWithAnswerTypeMidDim as BertFor2PairSequenceWithAnswerType
from typing import List, Tuple, Dict


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    # relationEmbedding = RelationEmbedding()
    haveRels = {}
    noRels = {}
    def __init__(self, guid, text_a, text_b=None, label=None, entitys = None, rels=[],\
                    answerType: str = '', answerStr: str = ''):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.freebaseEntity = entitys
        self.freebaseRels = rels
        # self.relsId = InputExample.relation2id(self.freebaseRels)
        self.answerType = answerType
        self.answerStr = answerStr
        



class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, entitysId = [], relsId = [],\
                        answerTypeIds:List[int] = [], answerStrIds:List[int] = []) -> None:
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.entitysId = entitysId
        self.relsId = relsId
        self.answerTypeIds = answerTypeIds
        self.answerStrIds = answerStrIds

class DataProcessor(object):
    "输入是来自文件中的[lable----que----path]的list"
    "输出是一个包含[id----lable----que_path]的list"
    def __init__(self, args):
        self.args = args

    def get_train_examples(self, data_dir):
        """See base class."""
        # logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train_rel_sim_<E>.txt")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, self.args.T_file_name)), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, self.args.v_file_name)), "dev")
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, self.args.t_file_name)), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines: List[List[str]], set_type: str):
        """Creates examples for the training and dev sets."""
        examples_all = []
        examples = []
        group_id = -1
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            line = eval(line)
            # import pdb; pdb.set_trace()
            text_a = line[0]
            qg = eval(line[1])
            # text_b = '\t'.join(line[2:7])
            # import pdb; pdb.set_trace()
            text_b = '\t'.join([''.join(qg['basePath']).lower(),\
                                ''.join(qg['higherOrderConstrain']).lower(),\
                                ''.join(qg['virtualConstrain']).lower(),\
                                ''.join(qg['entityPath']).lower(),\
                                ''.join(qg['relConstrain']).lower(),
                                ''.join(qg['mainPath']).lower()])
            label = qg["label"]
            answerType = qg["answerType"]
            # answerStr = line[10].lower()
            # import pdb; pdb.set_trace()
            if((i + 1) % self.args.group_size == 0):# 表示开始新的一组数据
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, rels=[],\
                                answerType=answerType, answerStr=''))
                examples_all.append(examples)
                examples = []
                # if(i > 100):
                #     break
            else:# 表示是同一组数据，可以继续放在一起
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, rels=[],\
                                answerType=answerType, answerStr=''))
        if(len(examples) != 0):
            examples_all.append(examples)
        # import pdb; pdb.set_trace()
        return examples_all


    def convert_examples_to_features(self, examples: List[List[str]],
                                 tokenizer) -> List[List[InputFeatures]]:
        """Loads a data file into a list of `InputBatch`s."""
        max_seq_length = self.args.max_seq_length
        features_all = []
        for (ex_index, example_group) in enumerate(examples):
            # print(ex_index)
            features = []
            if ex_index % 100000 == 0:
                print("Writing example %d of %d" % (ex_index, len(examples)))
            for example in example_group:
                tokens_a = tokenizer.tokenize(example.text_a)
                tokens_b = None
                if example.text_b:
                    text_b_list = example.text_b.split('\t')
                    tokens_b = []
                    for i, text_b in enumerate(text_b_list):
                        tokens_b += tokenizer.tokenize(text_b)
                        tokens_b.append('[unused' + str(i + 1) + ']')
                    self.truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
                else:
                    if len(tokens_a) > max_seq_length - 2:
                        tokens_a = tokens_a[:(max_seq_length - 2)]
                tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
                segment_ids = [0] * len(tokens)
                if tokens_b:
                    tokens += tokens_b + ["[SEP]"]
                    segment_ids += [1] * (len(tokens_b) + 1)     
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)
                # Zero-pad up to the sequence length.
                padding = [0] * (max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                # import pdb; pdb.set_trace()
                label_id = int(example.label)
                features.append(
                        InputFeatures(input_ids=input_ids,
                                    input_mask=input_mask,
                                    segment_ids=segment_ids,
                                    label_id=label_id,
                                    relsId=[]))
            features_all.append(features)
        return features_all

    def convert_examples_to_features_with_relsId(self, examples: List[List[str]],
                                 tokenizer) -> List[List[InputFeatures]]:
        """Loads a data file into a list of `InputBatch`s."""
        max_seq_length = self.args.max_seq_length
        features_all = []
        for (ex_index, example_group) in enumerate(examples):
            # print(ex_index)
            features = []
            if ex_index % 100000 == 0:
                print("Writing example %d of %d" % (ex_index, len(examples)))
            for example in example_group:
                tokens_a = tokenizer.tokenize(example.text_a)
                tokens_b = None
                if example.text_b:
                    text_b_list = example.text_b.split('\t')
                    tokens_b = []
                    for i, text_b in enumerate(text_b_list):
                        tokens_b += tokenizer.tokenize(text_b)
                        tokens_b.append('[unused' + str(i) + ']')
                    self.truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
                else:
                    if len(tokens_a) > max_seq_length - 2:
                        tokens_a = tokens_a[:(max_seq_length - 2)]
                tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
                segment_ids = [0] * len(tokens)
                if tokens_b:
                    tokens += tokens_b + ["[SEP]"]
                    segment_ids += [1] * (len(tokens_b) + 1)     
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)
                # Zero-pad up to the sequence length.
                padding = [0] * (max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                # import pdb; pdb.set_trace()
                label_id = int(example.label)
                features.append(
                        InputFeatures(input_ids=input_ids,
                                    input_mask=input_mask,
                                    segment_ids=segment_ids,
                                    label_id=label_id,
                                    relsId=example.relsId))
                bert_input = self.convert_sentence_to_bert_input(example.text_a, tokenizer)
                features.append(
                        InputFeatures(input_ids=bert_input[0],
                                    input_mask=bert_input[1],
                                    segment_ids=bert_input[2],
                                    label_id=label_id,
                                    relsId = example.relsId))
            features_all.append(features)
        return features_all

    def convert_sentence_pair_to_features(self, sentenceA: str, sentenceB: str, tokenizer):
        max_seq_length = self.args.max_seq_length
        tokens_a = tokenizer.tokenize(sentenceA)
        text_b_list = sentenceB.split('\t')
        tokens_b = []
        for i, text_b in enumerate(text_b_list):
            tokens_b += tokenizer.tokenize(text_b)
            tokens_b.append('[unused' + str(i + 1) + ']')
        self.truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)     
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        # import pdb; pdb.set_trace()
        return (input_ids, input_mask, segment_ids)


    def convert_examples_to_features_for_combine_answer(self, examples: List[List[str]],
                                 tokenizer) -> List[List[InputFeatures]]:
        """Loads a data file into a list of `InputBatch`s."""
        features_all = []
        for exampleGroup in examples:
            features = []
            for example in exampleGroup:
                # import pdb; pdb.set_trace()
                label_id = int(example.label)
                bertInput = self.convert_sentence_pair_to_features(example.text_a, example.text_b, tokenizer)
                features.append(
                        InputFeatures(input_ids=bertInput[0],
                                    input_mask=bertInput[1],
                                    segment_ids=bertInput[2],
                                    label_id=label_id))
                # text_a = ' '.join(example.text_a.split(' ')[0:2])
                text_b = example.answerType + '\t' + example.answerStr
                bert_input = self.convert_sentence_pair_to_features(example.text_a, text_b, tokenizer)
                features.append(
                        InputFeatures(input_ids=bert_input[0],
                                    input_mask=bert_input[1],
                                    segment_ids=bert_input[2],
                                    label_id=label_id))
            features_all.append(features)
        return features_all

        

    def convert_examples_to_features_with_answer_type(self, examples: List[List[str]],
                                 tokenizer) -> List[List[InputFeatures]]:
        """Loads a data file into a list of `InputBatch`s."""
        features_all = []
        for exampleGroup in examples:
            features = []
            for example in exampleGroup:
                # import pdb; pdb.set_trace()
                label_id = int(example.label)
                bertInput = self.convert_sentence_pair_to_features(example.text_a, example.text_b, tokenizer)
                features.append(
                        InputFeatures(input_ids=bertInput[0],
                                    input_mask=bertInput[1],
                                    segment_ids=bertInput[2],
                                    label_id=label_id))
                # text_a = ' '.join(example.text_a.split(' ')[0:2])
                # print(example.answerType)
                # import pdb; pdb.set_trace()
                bert_input = self.convert_sentence_pair_to_features(example.text_a, example.answerType, tokenizer)
                # bert_input = self.convert_sentence_pair_to_features(text_a, example.answerType, tokenizer)
                # import pdb; pdb.set_trace()
                # bert_input = self.convert_sentence_pair_to_features(example.text_a, example.answerStr, tokenizer)
                features.append(
                        InputFeatures(input_ids=bert_input[0],
                                    input_mask=bert_input[1],
                                    segment_ids=bert_input[2],
                                    label_id=label_id))
            features_all.append(features)
        return features_all

    def convert_examples_to_features_with_answer_str(self, examples: List[List[str]],
                                 tokenizer) -> List[List[InputFeatures]]:
        """Loads a data file into a list of `InputBatch`s."""
        features_all = []
        for exampleGroup in examples:
            features = []
            for example in exampleGroup:
                # import pdb; pdb.set_trace()
                label_id = int(example.label)
                bertInput = self.convert_sentence_pair_to_features(example.text_a, example.text_b, tokenizer)
                features.append(
                        InputFeatures(input_ids=bertInput[0],
                                    input_mask=bertInput[1],
                                    segment_ids=bertInput[2],
                                    label_id=label_id))
                bert_input = self.convert_sentence_pair_to_features(example.text_a, example.answerStr, tokenizer)
                features.append(
                        InputFeatures(input_ids=bert_input[0],
                                    input_mask=bert_input[1],
                                    segment_ids=bert_input[2],
                                    label_id=label_id))
            features_all.append(features)
        return features_all

    def convert_examples_to_features_with_answer_info(self, examples: List[List[str]],
                                 tokenizer) -> List[List[InputFeatures]]:
        """Loads a data file into a list of `InputBatch`s."""
        features_all = []
        for exampleGroup in examples:
            features = []
            for example in exampleGroup:
                # import pdb; pdb.set_trace()
                label_id = int(example.label)
                bertInput = self.convert_sentence_pair_to_features(example.text_a, example.text_b, tokenizer)
                features.append(
                        InputFeatures(input_ids=bertInput[0],
                                    input_mask=bertInput[1],
                                    segment_ids=bertInput[2],
                                    label_id=label_id))
                bert_input = self.convert_sentence_pair_to_features(example.text_a, example.answerType, tokenizer)
                features.append(
                        InputFeatures(input_ids=bert_input[0],
                                    input_mask=bert_input[1],
                                    segment_ids=bert_input[2],
                                    label_id=label_id))
                bert_input = self.convert_sentence_pair_to_features(example.text_a, example.answerStr, tokenizer)
                features.append(
                        InputFeatures(input_ids=bert_input[0],
                                    input_mask=bert_input[1],
                                    segment_ids=bert_input[2],
                                    label_id=label_id))
            features_all.append(features)
        return features_all


    def convert_sentence_to_bert_input(self, sentence: str, tokenizer) -> Tuple[List[int]]:
        max_seq_length = self.args.max_seq_length
        tokens_a = tokenizer.tokenize(sentence)
        self.truncate_seq(tokens_a, max_seq_length - 2)
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        # import pdb; pdb.set_trace()
        return (input_ids, input_mask, segment_ids)


    def convert_sentenceb_to_bert_input(self, sentence: str, tokenizer) -> Tuple[List[int]]:
        max_seq_length = self.args.max_seq_length
        text_b_list = sentence.split('\t')
        tokens_b = []
        for i, text_b in enumerate(text_b_list):
            tokens_b += tokenizer.tokenize(text_b)
            # tokens_b.append('[unused' + str(i) + ']')
        # tokens_a = tokenizer.tokenize(sentence)
        self.truncate_seq(tokens_b, max_seq_length - 2)
        tokens = ["[CLS]"] + tokens_b + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        # import pdb; pdb.set_trace()
        return (input_ids, input_mask, segment_ids)


    def convert_examples_to_features_with_two_sentence(self, examples: List[List[str]],
                                 tokenizer, file_mode: str = 'T') -> List[List[InputFeatures]]:
        """Loads a data file into a list of `InputBatch`s."""
        features_all = []
        for (ex_index, example_group) in enumerate(examples):
            features = []
            if ex_index % 100000 == 0:
                print("Writing example %d of %d" % (ex_index, len(examples)))
            for example in example_group:
                label_id = int(example.label)
                bert_input = self.convert_sentence_to_bert_input(example.text_a, tokenizer)
                features.append(
                        InputFeatures(input_ids=bert_input[0],
                                    input_mask=bert_input[1],
                                    segment_ids=bert_input[2],
                                    label_id=label_id,
                                    relsId = example.relsId))
                bert_input = self.convert_sentenceb_to_bert_input(example.text_b, tokenizer)
                features.append(
                        InputFeatures(input_ids=bert_input[0],
                                    input_mask=bert_input[1],
                                    segment_ids=bert_input[2],
                                    label_id=label_id,
                                    relsId = example.relsId))  
                # import pdb; pdb.set_trace()
            features_all.append(features)
        return features_all

    def truncate_seq(self, tokens_a, max_length):
        while True:
            total_length = len(tokens_a)
            if total_length <= max_length:
                break
            tokens_a.pop()

    def truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    # 构建输入模型的数据
    def build_data_for_model(self, eval_features: List[List[InputFeatures]], tokenizer, device):
        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        all_label_ids = []
        all_rels_ids = []
        all_answer_type_ids = []
        for eval_feature in eval_features:
            for f in eval_feature:
                # import pdb; pdb.set_trace()
                all_input_ids.append(f.input_ids)
                all_input_mask.append(f.input_mask)
                all_segment_ids.append(f.segment_ids)
                all_label_ids.append(f.label_id)
                all_rels_ids.append(f.relsId)
                # all_answer_type_ids.append(f.answerTypeIds)
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long).to(device)
        all_input_mask = torch.tensor(all_input_mask, dtype=torch.long).to(device)
        all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long).to(device)
        all_label_ids = torch.tensor(all_label_ids, dtype=torch.long).to(device)
        # import pdb; pdb.set_trace()
        all_rels_ids = torch.tensor(all_rels_ids, dtype=torch.long).to(device)
        # import pdb; pdb.set_trace()
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_rels_ids)
        return eval_data

    def build_data_for_model_train(self, eval_features: List[List[InputFeatures]], tokenizer, device):
        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        all_label_ids = []
        all_rels_ids = []
        for eval_feature in eval_features:
            temp = []
            temp2 = []
            temp3 = []
            temp4 = []
            temp5 = []
            for f in eval_feature:
                temp.append(f.input_ids)
                temp2.append(f.input_mask)
                temp3.append(f.segment_ids)
                temp4.append(f.label_id)
                temp5.append(f.relsId)
            all_input_ids.append(temp)
            all_input_mask.append(temp2)
            all_segment_ids.append(temp3)
            all_label_ids.append(temp4)
            all_rels_ids.append(temp5)
            # import pdb; pdb.set_trace()
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long).to(device)          
        all_input_mask = torch.tensor(all_input_mask, dtype=torch.long).to(device)
        all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long).to(device)  
        all_label_ids = torch.tensor(all_label_ids, dtype=torch.long).to(device)
        all_rels_ids = torch.tensor(all_rels_ids, dtype=torch.long).to(device)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_rels_ids)
        return eval_data

    @classmethod
    def _read_tsv(cls, input_file: str, quotechar=None) -> List[List[str]]:
        """Reads a tab separated value file."""       
        with open(input_file, "r", encoding="utf-8") as f:
            # reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            # import pdb;pdb.set_trace()
            for line in f:
                # if sys.version_info[0] == 2:
                #     line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line.strip())
            return lines

def main(fout_res, args: ArgumentParser):
    best_model_dir_name = ''
    processor = DataProcessor(args)
    device = torch.device("cuda", 0)
    shutil.copy(__file__, args.output_dir + __file__)
    merge_mode = ['listwise']
    # import pdb; pdb.set_trace()
    tokenizer = BertTokenizer.from_pretrained(args.bert_vocab, do_lower_case=args.do_lower_case)
    # 构建验证集数据  
    eval_examples = processor.get_dev_examples(args.data_dir)
    # import pdb; pdb.set_trace()   
    # eval_data = processor.convert_examples_to_features(eval_examples, tokenizer)
    eval_data = processor.convert_examples_to_features_with_answer_type(eval_examples, tokenizer)
    eval_data = processor.build_data_for_model(eval_data, tokenizer, device)
    train_examples = processor.get_train_examples(args.data_dir)
    num_train_optimization_steps = math.ceil(math.ceil(len(train_examples) / args.train_batch_size)\
                                        / args.gradient_accumulation_steps) * args.num_train_epochs    
    # import pdb; pdb.set_trace()   
    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE))
    model = BertFor2PairSequenceWithAnswerType.from_pretrained(args.bert_model,cache_dir=cache_dir,num_labels=1)
    model.to(device)
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            warmup=args.warmup_proportion,
                            t_total=num_train_optimization_steps)
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    # import pdb; pdb.set_trace()
    # **************************
    if args.do_train:   
        i_train_step = 0
        # train_data = processor.convert_examples_to_features(train_examples, tokenizer)
        train_data = processor.convert_examples_to_features_with_answer_type(train_examples, tokenizer)
        train_data = processor.build_data_for_model_train(train_data, tokenizer, device)
        dev_acc = 0.0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            # train_sampler = SequentialSampler(train_data)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
            model.train()
            tr_loss = 0
            point_loss = 0
            pair_loss = 0
            list_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            n_batch_correct = 0
            len_train_data = 0
            crossLoss = torch.nn.CrossEntropyLoss()
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                i_train_step += 1
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, rels_ids = batch
                # import pdb; pdb.set_trace()
                input_ids = input_ids.to(device).view(-1, args.max_seq_length)
                input_mask = input_mask.to(device).view(-1, args.max_seq_length)
                segment_ids = segment_ids.to(device).view(-1, args.max_seq_length)
                label_ids = label_ids.to(device).view(-1)
                # define a new function to compute loss values for both output_modes
                # import pdb; pdb.set_trace()
                logits = model(input_ids, segment_ids, input_mask, labels=None)
                # import pdb;pdb.set_trace()
                loss_point = torch.tensor(0.0).to(device)
                loss_pair = 0.0
                loss_list = 0.0
                if 'listwise' in merge_mode:
                    label_ids = label_ids.view(-1,2)[:,0]
                    logits_que = torch.softmax(logits.view(-1, args.group_size), 1)
                    label_ids_que = label_ids.view(-1, args.group_size)
                    # import pdb; pdb.set_trace()
                    for i, que_item in enumerate(logits_que):
                        for j, item in enumerate(que_item):
                            if(label_ids_que[i][j] == 0):
                                if(item != 1):
                                    loss_list += torch.log(1 - item)
                            else:
                                if(item != 0):
                                    loss_list += torch.log(item)
                                # import pdb; pdb.set_trace()
                    loss_list = 0 - loss_list
                    list_loss += loss_list.item()
                # 计算评价函数
                true_pos = torch.max(logits.view(-1, args.group_size), 1)[1]
                label_ids_que = label_ids.view(-1, args.group_size)
                for i, item in enumerate(true_pos):
                    if(label_ids_que[i][item] == 1):
                        n_batch_correct += 1
                len_train_data += logits.view(-1, args.group_size).size(0) 
                try:
                    loss = loss_list
                    loss.backward()      
                except:
                    import pdb; pdb.set_trace()
                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:                   
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1    
            optimizer.step()
            optimizer.zero_grad()
            print('train_loss:', tr_loss)    
            fout_res.write('single loss:' + str(point_loss) + '\t' + str(pair_loss) + '\t' + str(list_loss) + '\n')  
            fout_res.write('train loss:' + str(tr_loss) + '\n')
            P_train = 1. * int(n_batch_correct) / len_train_data
            print("train_Accuracy-----------------------",P_train)
            fout_res.write('train accuracy:' + str(P_train) + '\n')
            F_dev = 0
            if args.do_eval:
                file_name1 = args.output_dir + 'prediction_valid'
                f_valid = open(file_name1, 'w', encoding='utf-8')
                # Run prediction for full data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
                model.eval()
                P_dev = 0
                for input_ids, input_mask, segment_ids, label_ids, rels_ids in tqdm(eval_dataloader, desc="Evaluating"):
                    input_ids = input_ids.to(device).view(-1, args.max_seq_length)
                    input_mask = input_mask.to(device).view(-1, args.max_seq_length)
                    segment_ids = segment_ids.to(device).view(-1, args.max_seq_length)
                    label_ids = label_ids.to(device).view(-1)
                    rels_ids = rels_ids.to(device).view(-1, 2)
                    with torch.no_grad():
                        logits = model(input_ids, segment_ids, input_mask, labels=None)    
                    # logits = torch.sigmoid(logits)
                    # import pdb; pdb.set_trace()  
                    for item in logits:
                        f_valid.write(str(float(item)) + '\n')
                f_valid.flush()
                # p, r, F_dev = cal_f1_with_position_compq(file_name1, args.data_dir + args.v_file_name, 'v', -3)
                p, r, F_dev = cal_f1(file_name1, args.data_dir + args.v_file_name, 'v', actual_num=0)
                fout_res.write(str(p) + '\t' + str(r) + '\t' + str(F_dev) + '\n')
                fout_res.flush()
            if(True):
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_dir = args.output_dir + str(P_train) + '_' + str(F_dev) + '_' + str(_)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                if(F_dev > dev_acc):
                    best_model_dir_name = output_dir
                    dev_acc = F_dev
                    print(best_model_dir_name)
                    # If we save using the predefined names, we can load using `from_pretrained`
                    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
                    output_config_file = os.path.join(output_dir, CONFIG_NAME)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save.config.to_json_file(output_config_file)
                    tokenizer.save_vocabulary(output_dir)
    return best_model_dir_name

def test(best_model_dir_name, fout_res, args):
    print('测试选用的模型是', best_model_dir_name)
    fout_res.write('测试选用的模型是:' + best_model_dir_name + '\n')
    processor = DataProcessor(args)
    device = torch.device("cuda", 0)
    merge_mode = ['pairwise']
    tokenizer = BertTokenizer.from_pretrained(best_model_dir_name, do_lower_case=args.do_lower_case)
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE))
    model = BertFor2PairSequenceWithAnswerType.from_pretrained(best_model_dir_name,cache_dir=cache_dir,num_labels=1)
    model.to(device)
    # 构建验证集数据  
    eval_examples = processor.get_test_examples(args.data_dir)
    # import pdb; pdb.set_trace()   
    # eval_data = processor.convert_examples_to_features(eval_examples, tokenizer)
    eval_data = processor.convert_examples_to_features_with_answer_type(eval_examples, tokenizer)
    eval_data = processor.build_data_for_model(eval_data, tokenizer, device)
    # import pdb; pdb.set_trace()
    file_name1 = args.output_dir + 'prediction_test'
    f_valid = open(file_name1, 'w', encoding='utf-8')
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    model.eval()
    P_dev = 0
    for input_ids, input_mask, segment_ids, label_ids, rels_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device).view(-1, args.max_seq_length)
        input_mask = input_mask.to(device).view(-1, args.max_seq_length)
        segment_ids = segment_ids.to(device).view(-1, args.max_seq_length)
        label_ids = label_ids.to(device).view(-1)
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)    
        # logits = torch.sigmoid(logits)
        # import pdb; pdb.set_trace()
        for item in logits:
            f_valid.write(str(float(item)) + '\n')
    f_valid.flush()
    # p, r, F_dev = cal_f1_with_position_compq(file_name1, args.data_dir + args.t_file_name, 't', -3)
    p, r, F_dev = cal_f1(file_name1, args.data_dir + args.t_file_name, 't', actual_num = 0)
    fout_res.write(str(p) + '\t' + str(r) + '\t' + str(F_dev) + '\n')
    fout_res.flush()

if __name__ == "__main__":
    seed = 43
    steps = 1
    
    for steps in [100]:
        train_batch_size = 1
        logger = logging.getLogger(__name__)
        parser = ArgumentParser(description = 'For KBQA')
        parser.add_argument("--config_file",default='',type=str)
        # 加载yaml
        config = yaml.safe_load(open(parser.parse_args().config_file,'r'))
        N = config['N']
        parser.add_argument("--bert_model", default=BASE_DIR + '/bert_base_chinese', type=str)
        parser.add_argument("--bert_vocab", default=BASE_DIR + '/bert_base_chinese', type=str)
        parser.add_argument("--task_name",default='mrpc',type=str,help="The name of the task to train.")

        parser.add_argument("--data_dir",default=BASE_DIR + config['data_dir'],type=str)
        parser.add_argument("--output_dir",default=BASE_DIR + config['output_dir'],type=str)
        parser.add_argument("--T_file_name",default=config['T_file_name'],type=str)
        parser.add_argument("--v_file_name",default=config['v_file_name'],type=str)
        parser.add_argument("--t_file_name",default=config['t_file_name'],type=str)

        ## Other parameters
        parser.add_argument("--group_size",default=N,type=int,help="")
        parser.add_argument("--cache_dir",default="",type=str,help="Where do you want to store the pre-trained models downloaded from s3")
        parser.add_argument("--max_seq_length",default=150,type=int)
        parser.add_argument("--do_train",default='true',help="Whether to run training.")
        parser.add_argument("--do_eval",default='true',help="Whether to run eval on the dev set.")
        parser.add_argument("--do_lower_case",action='store_true',help="Set this flag if you are using an uncased model.")
        parser.add_argument("--train_batch_size",default=train_batch_size,type=int,help="Total batch size for training.")
        parser.add_argument("--eval_batch_size",default=128,type=int,help="Total batch size for eval.")
        parser.add_argument("--learning_rate",default=5e-5,type=float,help="The initial learning rate for Adam.")
        parser.add_argument("--num_train_epochs",default=5.0,type=float,help="Total number of training epochs to perform.")
        parser.add_argument("--warmup_proportion",default=0.1,type=float,)
        parser.add_argument("--no_cuda",action='store_true',help="Whether not to use CUDA when available")
        parser.add_argument("--local_rank",type=int,default=-1,help="local_rank for distributed training on gpus")
        parser.add_argument('--seed',type=int,default=seed,help="random seed for initialization")
        parser.add_argument('--gradient_accumulation_steps',type=int,default=steps,help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
        parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")  
        args = parser.parse_args()
        for i in args.__dict__:
            print('{}: {}'.format(i, args.__dict__[i]))

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)   
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu_id'])
        # exit()
        # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        fout_res = open(args.output_dir + 'result.log', 'a+', encoding='utf-8')
        best_model_dir_name = main(fout_res, args)

        # best_model_dir_name = '/home/chenwenliang/jiayonghui/ckbqa/src/rerank/model_train/models/rerank_ccks_listwise_bert_negOrder_answerType_seq150_CE_group_10_1_42_100/0.9946043165467626_0.9008977283268468_4'
        test(best_model_dir_name, fout_res, args)