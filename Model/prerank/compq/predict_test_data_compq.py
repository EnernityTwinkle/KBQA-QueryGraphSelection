from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import numpy as np
import torch
import math
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
# from model_BertForSequence import BertForSequenceClassificationAddScore
# from model_BertForSequence import FocalLoss
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(BASE_DIR)

from Model.cal_f1 import cal_f1, cal_f1_with_position, cal_f1_with_position_compq
import pickle
import shutil



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
class DataProcessor(object):
    "返回一个包含所有[lable----que----path]的list"
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""       
        with open(input_file, "r", encoding="utf-8") as f:
            # reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            # import pdb;pdb.set_trace()
            for line in f:
                # if sys.version_info[0] == 2:
                #     line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line.strip().split('\t'))
            return lines
class MrpcProcessor(DataProcessor):
    "输入是来自文件中的[lable----que----path]的list"
    "输出是一个包含[id----lable----que_path]的list"

    def get_train_examples(self, data_dir):
        """See base class."""
        # logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train_rel_sim_<E>.txt")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, args.T_file_name)), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, args.v_file_name)), "dev")
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, args.t_file_name)), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples_all = []
        examples = []
        group_id = -1
        for (i, line) in enumerate(lines):
            if((i + 1) % args.group_size == 0):# 表示开始新的一组数据
                # if(len(line) != 11):
                #     print(line)
                #     import pdb; pdb.set_trace()
                group_id = line[-1]
                guid = "%s-%s" % (set_type, i)
                text_a = line[1]
                text_b = '\t'.join(line[2:7])
                label = line[0]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                examples_all.append(examples)
                examples = []
                # if(i > 100):
                #     break
            else:# 表示是同一组数据，可以继续放在一起
                guid = "%s-%s" % (set_type, i)
                text_a = line[1]
                text_b = '\t'.join(line[2:7])
                label = line[0]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        # print(len(examples_all))
        # import pdb; pdb.set_trace()
        if(len(examples) != 0):
            examples_all.append(examples)
        return examples_all

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode, file_mode = 'T'):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {label : i for i, label in enumerate(label_list)}  
    features_all = []
    for (ex_index, example_group) in enumerate(examples):
        # print(ex_index)
        features = []
        if ex_index % 100000 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))
        # import pdb; pdb.set_trace()
        for example in example_group:
            tokens_a = tokenizer.tokenize(example.text_a)
            tokens_b = None
            if example.text_b:
                text_b_list = example.text_b.split('\t')
                tokens_b = []
                for i, text_b in enumerate(text_b_list):
                    tokens_b += tokenizer.tokenize(text_b)
                    tokens_b.append('[unused' + str(i) + ']')
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)     
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # import pdb; pdb.set_trace()
            input_mask = [1] * len(input_ids)
            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            if output_mode == "classification" or output_mode == "pairwise" or output_mode == "listwise":
                label_id = label_map[example.label]
            elif output_mode == "regression":
                label_id = float(example.label)
            else:
                raise KeyError(output_mode)
            features.append(
                    InputFeatures(input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                label_id=label_id))
        features_all.append(features)
    # import pdb; pdb.set_trace()
    return features_all


# 构建输入模型的数据
def build_data_for_model(examples, label_list, tokenizer, output_mode, device):
    eval_features = convert_examples_to_features(
        examples, label_list, args.max_seq_length, tokenizer, output_mode)
    all_input_ids = []
    for eval_feature in eval_features:
        for f in eval_feature:
            all_input_ids.append(f.input_ids)
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long).to(device)
    all_input_mask = []
    for eval_feature in eval_features:
        for f in eval_feature:
            all_input_mask.append(f.input_mask)
    all_input_mask = torch.tensor(all_input_mask, dtype=torch.long).to(device)
    all_segment_ids = []
    for eval_feature in eval_features:
        for f in eval_feature:
            all_segment_ids.append(f.segment_ids)
    all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long).to(device)
    all_label_ids = []
    for eval_feature in eval_features:
        for f in eval_feature:
            all_label_ids.append(f.label_id)
    all_label_ids = torch.tensor(all_label_ids, dtype=torch.long).to(device)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return eval_data

def build_data_for_model_train(examples, label_list, tokenizer, output_mode, device):
    eval_features = convert_examples_to_features(
            examples, label_list, args.max_seq_length, tokenizer, output_mode)
    all_input_ids = []
    for eval_feature in eval_features:
        temp = []
        for f in eval_feature:
            temp.append(f.input_ids)
        all_input_ids.append(temp)
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long).to(device)
    all_input_mask = []
    for eval_feature in eval_features:
        temp = []
        for f in eval_feature:
            temp.append(f.input_mask)
        all_input_mask.append(temp)
    all_input_mask = torch.tensor(all_input_mask, dtype=torch.long).to(device)
    all_segment_ids = []
    for eval_feature in eval_features:
        temp = []
        for f in eval_feature:
            temp.append(f.segment_ids)
        all_segment_ids.append(temp)
    all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long).to(device)
    all_label_ids = []
    for eval_feature in eval_features:
        temp = []
        for f in eval_feature:
            temp.append(f.label_id)
        all_label_ids.append(temp)
    all_label_ids = torch.tensor(all_label_ids, dtype=torch.long).to(device)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return eval_data


def test(best_model_dir_name, fout_res):
    print('测试选用的模型是', best_model_dir_name)
    fout_res.write('测试选用的模型是:' + best_model_dir_name + '\n')
    processors = {"mrpc": MrpcProcessor}
    output_modes = {"mrpc": "classification"}
    # output_modes = {"mrpc": "listwise"}
    device = torch.device("cuda", 0)
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    if(output_modes['mrpc'] == 'listwise'):
        num_labels = 1
    num_labels = 1
    tokenizer = BertTokenizer.from_pretrained(best_model_dir_name, do_lower_case=args.do_lower_case)
    # tokenizer.add_tokens('<E>')
    # tokenizer.add_special_tokens('<E>')
    train_examples = None
    num_train_optimization_steps = None 
    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE))
    # model = BertForSequenceClassification.from_pretrained(args.output_dir + args.input_model_dir,cache_dir=cache_dir,num_labels=1)
    model = BertForSequenceClassification.from_pretrained(best_model_dir_name,cache_dir=cache_dir,num_labels=num_labels)
    model.to(device)
    # Prepare optimizer
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    # 构建验证集数据
    # test_pickle = open(args.t_model_data_name, 'rb')
    # eval_data = pickle.load(test_pickle)
    eval_examples = processor.get_train_examples(args.data_dir)
    eval_data = build_data_for_model(eval_examples, label_list, tokenizer, output_mode, device)
    # ************************** 
    file_name1 = args.output_dir + 'prediction_test'
    f = open(file_name1, 'w', encoding='utf-8')
    if args.do_eval:
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        model.eval()
        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device).view(-1, args.max_seq_length)
            input_mask = input_mask.to(device).view(-1, args.max_seq_length)
            segment_ids = segment_ids.to(device).view(-1, args.max_seq_length)
            # import pdb; pdb.set_trace()
            label_ids = label_ids.to(device).view(-1)
            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)  
            logits = torch.sigmoid(logits)
            # create eval loss and other metric required by the task
            for item in logits:
                if(num_labels == 2):
                    f.write(str(float(item[1])) + '\n')
                else:
                    f.write(str(float(item)) + '\n')
        f.flush()
        p, r, f = cal_f1(file_name1, args.data_dir + args.T_file_name, 'T', actual_num = 1)
        fout_res.write('precision:' + str(p) + '\trecall:' + str(r) + '\tf1:' + str(f) + '\n')
        fout_res.flush()

if __name__ == "__main__":
    # for seed in [42, 1000, 10000]:
    dirName = BASE_DIR + '/runnings/model/compq/'
    # model2file = {'bert_compq_cross_valid0140_42_100/0.8123515439429929_0.2677426321121402_4':'compq_pointwise_all_train.txt1',
    #                 'bert_compq_cross_valid1140_42_100/0.7205146533238027_0.19539581986640805_4':'compq_pointwise_all_train.txt0'}
    # model2file = {'3711_bert_compq_pointwise_withans_rank1_f01_gradual_merge_type_entity_time_ordianl_mainpath_neg_20_42_100/0.9259954921111946_0.3759301486616067_2':'compq_test_all.txt'}
    model2file = {'3711_bert_compq_pointwise_withans_rank1_f01_gradual_merge_type_entity_time_ordianl_mainpath_neg_20_42_100/0.9259954921111946_0.3759301486616067_2':'compq_dev_all.txt'}
    for model in model2file:
        for seed in [42]:
            for steps in[100]:
                for N in [40]:
                    steps = int(800 / N)
                    logger = logging.getLogger(__name__)
                    print(seed)
                    os.environ["CUDA_VISIBLE_DEVICES"] = '4'
                    parser = ArgumentParser(description = 'For KBQA')
                    parser.add_argument("--data_dir",default=BASE_DIR + '/runnings/train_data/compq/',type=str)
                    parser.add_argument("--bert_model", default='/home/jiayonghui/github/bert_rank_data/bert_base_uncased', type=str)
                    parser.add_argument("--bert_vocab", default='/home/jiayonghui/github/bert_rank_data/bert_base_uncased', type=str)
                    parser.add_argument("--task_name",default='mrpc',type=str,help="The name of the task to train.")
                    parser.add_argument("--output_dir",default='/'.join((dirName + model).split('/')[0:-1]) + '/',type=str)
                    parser.add_argument("--input_model_dir", default='0.9675389502344577_0.4803025192052977_3', type=str)
                    parser.add_argument("--T_file_name",default=model2file[model],type=str)
                    parser.add_argument("--T_model_data_name",default='train_all_518484_from_1_500000000.pkl',type=str)
                    parser.add_argument("--v_model_data_name",default='dev_all_135428_from_v_bert_rel_answer_pairwise_1_500000000.pkl',type=str)
                    parser.add_argument("--t_model_data_name",default='test_all_344985_from_1_500000000.pkl',type=str)
                    ## Other parameters
                    parser.add_argument("--group_size",default=N + 1,type=int,help="")
                    parser.add_argument("--cache_dir",default="",type=str,help="Where do you want to store the pre-trained models downloaded from s3")
                    parser.add_argument("--max_seq_length",default=100,type=int)
                    parser.add_argument("--do_train",default='true',help="Whether to run training.")
                    parser.add_argument("--do_eval",default='true',help="Whether to run eval on the dev set.")
                    parser.add_argument("--do_lower_case",action='store_true',help="Set this flag if you are using an uncased model.")
                    parser.add_argument("--train_batch_size",default=1,type=int,help="Total batch size for training.")
                    parser.add_argument("--eval_batch_size",default=100,type=int,help="Total batch size for eval.")
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
                    random.seed(args.seed)
                    np.random.seed(args.seed)
                    torch.manual_seed(args.seed)
                    torch.cuda.manual_seed_all(args.seed)   
                    
                    fout_res = open(args.output_dir + 'result_predict_train.log', 'w', encoding='utf-8')
                    best_model_dir_name = dirName + model
                    test(best_model_dir_name, fout_res)