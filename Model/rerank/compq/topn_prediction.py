import argparse
import csv
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
from Model.common.InputExample import InputExample
from Model.cal_f1 import cal_f1, cal_f1_with_position, cal_f1_with_position_compq
from Model.common.DataProcessor import DataProcessor
# from Model.common.BertEncoderX import BertFor2PairSequenceWithAnswerType
from Model.common.BertEncoderX import BertFor2PairSequenceWithAnswerTypeMidDim as BertFor2PairSequenceWithAnswerType
from Model.rerank.loss import crossEntropy

    

def main(fout_res, args: ArgumentParser):
    best_model_dir_name = ''
    processor = DataProcessor(args)
    device = torch.device("cuda", 0)
    shutil.copy(__file__, args.output_dir + __file__)
    merge_mode = ['classification']
    # import pdb; pdb.set_trace()
    tokenizer = BertTokenizer.from_pretrained(args.bert_vocab, do_lower_case=args.do_lower_case)
    # 构建验证集数据  
    eval_examples = processor.get_dev_examples(args.data_dir)
    # import pdb; pdb.set_trace()   
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
                if "classification" in merge_mode:
                    # *******************交叉熵损失函数*********************
                    label_ids = label_ids.view(-1,2)[:,0]
                    loss_point = crossEntropy(logits, label_ids)
                    n_batch_correct += torch.sum(torch.eq((logits>0.5).data.long().view(-1,1),label_ids.view(-1,1)))
                    len_train_data += logits.size(0)   
                # import pdb;pdb.set_trace()
                try:
                    loss = loss_point
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
                    # logits = torch.softmax(logits, 1)    
                    # import pdb; pdb.set_trace()  
                    for item in logits:
                        f_valid.write(str(float(item)) + '\n')
                f_valid.flush()
                p, r, F_dev = cal_f1_with_position_compq(file_name1, args.data_dir + args.v_file_name, 'v', -3)
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
    eval_data = processor.convert_examples_to_features_with_answer_type(eval_examples, tokenizer)
    eval_data = processor.build_data_for_model(eval_data, tokenizer, device)
    # import pdb; pdb.set_trace()
    file_name1 = args.output_dir + 'prediction_test_topn_prediction'
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
            logits = model(input_ids, segment_ids, input_mask, labels=None, rels_ids = rels_ids)    
        # logits = torch.softmax(logits, 1)     
        # import pdb; pdb.set_trace()
        for item in logits:
            f_valid.write(str(float(item)) + '\n')
    f_valid.flush()
    p, r, F_dev = cal_f1_with_position_compq(file_name1, args.data_dir + args.t_file_name, 't', -3)
    fout_res.write(str(p) + '\t' + str(r) + '\t' + str(F_dev) + '\n')
    fout_res.flush()

if __name__ == "__main__":
    seed = 42
    steps = 50
    # for N in [5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 140]:
    # for N in [5, 10, 20, 30]:
    # for N in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]:
    fout_res = open('/home/jiayonghui/github/sum/RankingQueryGraphs/runnings/model/compq/compq_rerank_2bert_answer_type_pointwise_to2add_neg_40_42_50/' +\
                         'result_topn_prediction_2.log', 'w', encoding='utf-8')
    for N in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        logger = logging.getLogger(__name__)
        print(seed, N)
        os.environ["CUDA_VISIBLE_DEVICES"] = '5'
        parser = ArgumentParser(description = 'For KBQA')
        parser.add_argument("--data_dir",default=BASE_DIR + '/runnings/train_data/compq/',type=str)
        # parser.add_argument("--bert_model", default='bert-base-uncased', type=str)
        # parser.add_argument("--bert_vocab", default='bert-base-uncased', type=str)
        parser.add_argument("--bert_model", default= '/home/jiayonghui/github/bert_rank_data/bert_base_uncased', type=str)
        parser.add_argument("--bert_vocab", default='/home/jiayonghui/github/bert_rank_data/bert_base_uncased', type=str)
        parser.add_argument("--task_name",default='mrpc',type=str,help="The name of the task to train.")
        # parser.add_argument("--output_dir",default=BASE_DIR + '/runnings/model/webq/only_sim_no_answer_str_cat_2bert_group1_webq_pointwise_2linear_neg_' + str(N) + '_' + str(seed) + '_' + str(steps) + '/',type=str)
        parser.add_argument("--output_dir",default=BASE_DIR + '/runnings/model/compq/compq_rerank_2bert_answer_type_pointwise_to2add_neg_40' + '_' + str(seed) + '_' + str(steps) + '/',type=str)
        parser.add_argument("--input_model_dir", default='0.9675389502344577_0.4803025192052977_3', type=str)
        parser.add_argument("--T_file_name",default='compq_T_bert_2cv_constrain_top' + str(N) + '.txt',type=str)
        # parser.add_argument("--v_file_name",default='pairwise_with_freebase_id_dev_all_cut.txt',type=str)
        parser.add_argument("--v_file_name",default='compq_v_top' + str(N) + '.txt',type=str)
        # parser.add_argument("--v_file_name",default='webq_rank1_f01_gradual_label_position_1_' + str(N) + '_type_entity_time_ordinal_mainpath_is_train.txt',type=str)
        parser.add_argument("--t_file_name",default='compq_t_top' + str(N) + '.txt',type=str)

        parser.add_argument("--T_model_data_name",default='train_all_518484_from_1_500000000.pkl',type=str)
        parser.add_argument("--v_model_data_name",default='dev_all_135428_from_v_bert_rel_answer_pairwise_1_500000000.pkl',type=str)
        parser.add_argument("--t_model_data_name",default='test_all_344985_from_1_500000000.pkl',type=str)
        ## Other parameters
        parser.add_argument("--group_size",default=1,type=int,help="")
        parser.add_argument("--cache_dir",default="",type=str,help="Where do you want to store the pre-trained models downloaded from s3")
        parser.add_argument("--max_seq_length",default=100,type=int)
        parser.add_argument("--do_train",default='true',help="Whether to run training.")
        parser.add_argument("--do_eval",default='true',help="Whether to run eval on the dev set.")
        parser.add_argument("--do_lower_case",default='True', action='store_true',help="Set this flag if you are using an uncased model.")
        parser.add_argument("--train_batch_size",default=16,type=int,help="Total batch size for training.")
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

        best_model_dir_name = '/home/jiayonghui/github/sum/RankingQueryGraphs/runnings/model/compq/compq_rerank_2bert_answer_type_pointwise_to2add_neg_40_42_50/0.991625_0.4380796003411835_3/'
        fout_res.write('This is Top-' + str(N) + '\n')
        test(best_model_dir_name, fout_res, args)
        