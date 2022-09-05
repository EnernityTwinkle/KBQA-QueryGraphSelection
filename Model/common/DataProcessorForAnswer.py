import sys
import os
import torch
from typing import List, Dict, Tuple
from Model.common.InputExample import InputExample
from Model.common.InputFeatures import InputFeatures
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)


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

    def _create_examples(self, lines: List[List[str]], set_type: str) -> List[List[InputExample]]:
        """Creates examples for the training and dev sets."""
        examples_all = []
        examples = []
        group_id = -1
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            text_b = '\t'.join(line[2:3])
            label = line[0]
            # import pdb; pdb.set_trace()
            if((i + 1) % self.args.group_size == 0):# 表示开始新的一组数据
                if(i < 2):
                    print(line)
                    print(text_a, text_b)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                examples_all.append(examples)
                examples = []
                # if(i > 100):
                #     break
            else:# 表示是同一组数据，可以继续放在一起
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        if(len(examples) != 0):
            examples_all.append(examples)
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
            tokens_b.append('[unused' + str(i) + ']')
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
                # bert_input = self.convert_sentence_pair_to_features(example.text_a, example.answerType, tokenizer)
                # bert_input = self.convert_sentence_pair_to_features(text_a, example.answerType, tokenizer)
                # import pdb; pdb.set_trace()
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
                lines.append(line.strip().split('\t'))
            return lines
