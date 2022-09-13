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
from pytorch_pretrained_bert.modeling import *
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from Model.Pairwise.Embedding import RelationEmbedding
from typing import List, Dict, Tuple


class TwoBertForTwoSequence(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(TwoBertForTwoSequence, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.bert2 = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.classifier_2 = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier_3 = nn.Linear(config.hidden_size * 3, num_labels)
        self.denseCat = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        max_seq_length = 100
        que_input_ids = input_ids.view(-1, 2, max_seq_length)[:,0,:].view(-1, max_seq_length)
        graph_input_ids = input_ids.view(-1, 2, max_seq_length)[:,1,:].view(-1, max_seq_length)
        que_token_type_ids = token_type_ids.view(-1, 2, max_seq_length)[:,0,:].view(-1, max_seq_length)
        graph_token_type_ids = token_type_ids.view(-1, 2, max_seq_length)[:,1,:].view(-1, max_seq_length)
        que_attention_mask = attention_mask.view(-1, 2, max_seq_length)[:, 0, :].view(-1, max_seq_length)
        graph_attention_mask = attention_mask.view(-1, 2, max_seq_length)[:,1,:].view(-1, max_seq_length)
        _, pooled_output1 = self.bert(que_input_ids, que_token_type_ids, que_attention_mask, output_all_encoded_layers=False)
        _, pooled_output2 = self.bert2(graph_input_ids, graph_token_type_ids, graph_attention_mask, output_all_encoded_layers=False)
        # num_sen = pooled_output.shape[0]
        pooled_output = torch.cat((pooled_output1,pooled_output2),1)
        # import pdb; pdb.set_trace()
        ##########################################################
        denseCat = self.denseCat(pooled_output)
        denseCat = self.activation(denseCat)
        pooled_output = self.dropout(denseCat)
        logits = self.classifier(pooled_output)
        return logits

class BertForTwoSequence(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForTwoSequence, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.classifier_2 = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier_3 = nn.Linear(config.hidden_size * 3, num_labels)
        self.denseCat = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.dense3 = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.activation = nn.Tanh()
        # self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        num_sen = pooled_output.shape[0]
        
        # # ##################################两个向量相减的结果作为相似度特征************
        # # try:
        # #     for i in range(0, num_sen, 2):
        # #         cat_torch[i // 2] = pooled_output[i] - pooled_output[i + 1]
        # # except:
        # #     import pdb; pdb.set_trace()
        # # ##########################################################
        # ##################################两个向量拼接以及相减的结果作为相似度特征************
        # cat_torch = torch.rand((num_sen // 2, 768 * 3), device='cuda:0')
        # for i in range(0, num_sen, 2):
        #     catTensor = torch.cat((pooled_output[i], pooled_output[i+1]), 0)
        #     subTensor = pooled_output[i] - pooled_output[i + 1]
        #     cat_torch[i // 2] = torch.cat((catTensor, subTensor), 0)
        # denseCat = self.dense3(cat_torch)
        # denseCat = self.activation(denseCat)
        # pooled_output = self.dropout(denseCat)
        # logits = self.classifier(pooled_output)
                # import pdb; pdb.set_trace()
        ##########################################################
        ######两个向量拼接并经过dense和激活函数映射为768维，再进行分类########
        denseCat = self.denseCat(pooled_output.view(-1, 2 * 768)) 
        denseCat = self.activation(denseCat)
        # denseCat = pooled_output.view(-1, 2 * 768)
        pooled_output = self.dropout(denseCat)
        logits = self.classifier(pooled_output)
        # ************************************************
        return logits


class BertForSequence(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForSequence, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.classifier_2 = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier_3 = nn.Linear(config.hidden_size * 3, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class BertForTwoSequenceCosine(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForTwoSequenceCosine, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.classifier_2 = nn.Linear(config.hidden_size, num_labels)
        self.classifier_3 = nn.Linear(config.hidden_size * 3, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        num_sen = pooled_output.shape[0]
        cat_torch = torch.rand((num_sen // 2, 768 * 3), device='cuda:0')
        logits = torch.rand((num_sen // 2), device='cuda:0')
        ########################计算余弦相似度###################################
        try:
            for i in range(0, num_sen, 2):
                logits[i // 2] = torch.cosine_similarity(pooled_output[i], pooled_output[i + 1],dim=0)
                # import pdb; pdb.set_trace()
        except:
            import pdb; pdb.set_trace()
        ##########################################################
        # pooled_output = self.dropout(cat_torch)
        # logits = self.classifier_3(pooled_output).view(num_sen // 2, -1)
        # import pdb; pdb.set_trace()
        return logits


class BertForSequence(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForSequence, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # num_sen = pooled_output.shape[0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class BertForSequenceWithRels(BertPreTrainedModel):

    def __init__(self, config, num_labels):
        super(BertForSequenceWithRels, self).__init__(config)
        self.num_labels = num_labels
        # import pdb; pdb.set_trace()
        self.relEmbedding = torch.from_numpy(RelationEmbedding().embedding.astype(np.float32))
        self.relEmbeddingMatrix = torch.nn.Embedding.from_pretrained(self.relEmbedding, freeze=False)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.classifier_2 = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier_3 = nn.Linear(config.hidden_size * 3, num_labels)
        self.classifier_transe = nn.Linear(config.hidden_size * 2 + 50, num_labels)
        self.classifier_transe2 = nn.Linear(config.hidden_size + 50, 300)
        self.classifier_base_transe = nn.Linear(config.hidden_size + 300, num_labels)
        self.activation = nn.Tanh()
        self.apply(self.init_bert_weights)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, rels_ids = None):
        # import pdb; pdb.set_trace()
        #***********************使用transe直接拼接***********************
        rels_ids = rels_ids.view(-1,2,2)[:,0].view(-1,2)
        rels_emb = self.relEmbeddingMatrix(rels_ids)
        rels_emb = rels_emb.permute(0, 2, 1)
        rels_emb = torch.nn.functional.avg_pool1d(rels_emb, kernel_size=rels_emb.shape[-1]).squeeze(-1)
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = pooled_output.view(-1, 2 * 768)
        pooled_output = torch.cat((pooled_output, rels_emb), 1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier_transe(pooled_output)
        #***********************使用transe先映射为特征再拼接***********************
        # rels_ids = rels_ids.view(-1,2,2)[:,0].view(-1,2)
        # rels_emb = self.relEmbeddingMatrix(rels_ids)
        # rels_emb = rels_emb.permute(0, 2, 1)
        # rels_emb = torch.nn.functional.avg_pool1d(rels_emb, kernel_size=rels_emb.shape[-1]).squeeze(-1)
        # _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # pooled_output1 = pooled_output.view(-1, 2, 768)[:,0,:].view(-1, 768)
        # pooled_output2 = pooled_output.view(-1, 2, 768)[:,1,:].view(-1, 768)
        # pooled_output_transe = torch.cat((pooled_output2, rels_emb), 1)
        # pooled_output_transe = self.classifier_transe2(pooled_output_transe)
        # pooled_output_transe = self.activation(pooled_output_transe)
        # baseCatTranse = torch.cat((pooled_output1, pooled_output_transe), 1)
        # pooled_output = self.dropout(baseCatTranse)
        # logits = self.classifier_base_transe(pooled_output)
        ######################不使用transe###############################
        # _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # pooled_output = pooled_output.view(-1, 2, 768)[:,0,:].view(-1, 768)
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        # import pdb; pdb.set_trace()
        return logits

    
class BertForSequenceWithAnswerType(BertPreTrainedModel):

    def __init__(self, config, num_labels, mid_dim = 768):
        super(BertForSequenceWithAnswerType, self).__init__(config)
        self.num_labels = num_labels
        self.mid_dim = mid_dim
        # import pdb; pdb.set_trace()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.denseCat = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.apply(self.init_bert_weights)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, rels_ids = None):
        ######################问句与答案相似度和语义相似度拼接###############################
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        denseCat = self.denseCat(pooled_output.view(-1, 2 * 768)) 
        denseCat = self.activation(denseCat)
        pooled_output = self.dropout(denseCat)
        logits = self.classifier(pooled_output)
        ###############问句与答案相似度和语义相似度得分相加########################
        # _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        # logits = torch.sum(logits.view(-1, 2, 2),1)
        # import pdb; pdb.set_trace()
        ##########不使用answer信息#############################
        # input_ids = input_ids.view(-1, 2, 100)
        # token_type_ids = token_type_ids.view(-1, 2, 100)
        # attention_mask = attention_mask.view(-1, 2, 100)
        # input_ids1 = input_ids[:, 0, :]
        # token_type_ids1 = token_type_ids[:, 0, :]
        # attention_mask1 = attention_mask[:, 0, :]
        # _, pooled_output = self.bert(input_ids1, token_type_ids1, attention_mask1)
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        return logits


class BertFor2PairSequenceWithAnswerType(BertPreTrainedModel):
    
    def __init__(self, config, num_labels):
        super(BertFor2PairSequenceWithAnswerType, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.bert2 = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.denseCat = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, rels_ids = None):
        ##############问句与答案字符串的编码和语义相似度编码不采用同一个bert##############
        input_ids = input_ids.view(-1, 2, 100)
        token_type_ids = token_type_ids.view(-1, 2, 100)
        attention_mask = attention_mask.view(-1, 2, 100)
        input_ids1 = input_ids[:, 0, :]
        input_ids2 = input_ids[:, 1, :]
        token_type_ids1 = token_type_ids[:, 0, :]
        token_type_ids2 = token_type_ids[:, 1, :]
        attention_mask1 = attention_mask[:, 0, :]
        attention_mask2 = attention_mask[:, 1, :]
        _, pooled_output1 = self.bert(input_ids1, token_type_ids1, attention_mask1)
        _, pooled_output2 = self.bert2(input_ids2, token_type_ids2, attention_mask2)
        pooled_output = torch.cat((pooled_output1,pooled_output2),1)
        denseCat = self.denseCat(pooled_output) 
        denseCat = self.activation(denseCat)
        pooled_output = self.dropout(denseCat)
        logits = self.classifier(pooled_output)
        return logits

class BertFor2PairSequenceWithAnswerTypeMidDimPoint(BertPreTrainedModel):
# Pointwise方法试验
    
    def __init__(self, config, num_labels, mid_dim = 768):
        super(BertFor2PairSequenceWithAnswerTypeMidDimPoint, self).__init__(config)
        self.num_labels = num_labels
        self.mid_dim = mid_dim
        self.bert = BertModel(config)
        self.bert2 = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier1 = nn.Linear(config.hidden_size, num_labels)
        self.classifier2 = nn.Linear(config.hidden_size, num_labels)
        # self.denseCat = nn.Linear(config.hidden_size * 2, )
        # self.activation = nn.Tanh()
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, rels_ids = None):
        input_ids = input_ids.view(-1, 2, 100)
        token_type_ids = token_type_ids.view(-1, 2, 100)
        attention_mask = attention_mask.view(-1, 2, 100)
        input_ids1 = input_ids[:, 0, :]
        input_ids2 = input_ids[:, 1, :]
        token_type_ids1 = token_type_ids[:, 0, :]
        token_type_ids2 = token_type_ids[:, 1, :]
        attention_mask1 = attention_mask[:, 0, :]
        attention_mask2 = attention_mask[:, 1, :]
        _, pooled_output1 = self.bert(input_ids1, token_type_ids1, attention_mask1)
        _, pooled_output2 = self.bert2(input_ids2, token_type_ids2, attention_mask2)
        ##############问句与答案字符串的编码和语义相似度编码不采用同一个bert,直接拼接映射到2维##############
        pooled_output = torch.cat((pooled_output1,pooled_output2),1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        ##############问句与答案字符串的编码和语义相似度编码不采用同一个bert,分别映射到2维再相加##############
        logits1 = self.classifier1(self.dropout(pooled_output1))
        logits2 = self.classifier2(self.dropout(pooled_output2))
        # logits = logits1 + logits2
        logits = logits1 * torch.sigmoid(logits2)
        # logits = logits1
        return logits

class BertFor2PairSequenceWithAnswerTypeMidDim(BertPreTrainedModel):
    
    def __init__(self, config, num_labels, mid_dim = 768):
        super(BertFor2PairSequenceWithAnswerTypeMidDim, self).__init__(config)
        # import pdb; pdb.set_trace()
        self.num_labels = num_labels
        self.mid_dim = mid_dim
        self.bert = BertModel(config)
        self.bert2 = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier1 = nn.Linear(config.hidden_size, num_labels)
        self.classifier2 = nn.Linear(config.hidden_size, num_labels)
        self.denseCat = nn.Linear(config.hidden_size * 2, 1)
        self.mlp1 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        # self.mlp2 = nn.Linear(config.hidden_size, num_labels)
        self.mlp2 = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.mlp3 = nn.Linear(config.hidden_size // 2, num_labels)
        #############
        # self.mlp1 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        # self.mlp2 = nn.Linear(config.hidden_size, config.hidden_size // 2)
        # self.mlp3 = nn.Linear(config.hidden_size // 2, config.hidden_size // 4)
        # self.mlp4 = nn.Linear(config.hidden_size // 4, num_labels)
        # self.activation = nn.Tanh()
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, rels_ids = None):
        input_ids = input_ids.view(-1, 2, 100)
        token_type_ids = token_type_ids.view(-1, 2, 100)
        attention_mask = attention_mask.view(-1, 2, 100)
        input_ids1 = input_ids[:, 0, :]
        input_ids2 = input_ids[:, 1, :]
        token_type_ids1 = token_type_ids[:, 0, :]
        token_type_ids2 = token_type_ids[:, 1, :]
        attention_mask1 = attention_mask[:, 0, :]
        attention_mask2 = attention_mask[:, 1, :]
        _, pooled_output1 = self.bert(input_ids1, token_type_ids1, attention_mask1)
        _, pooled_output2 = self.bert2(input_ids2, token_type_ids2, attention_mask2)
        ##############问句与答案字符串的编码和语义相似度编码不采用同一个bert,直接拼接映射到2维##############
        # pooled_output = torch.cat((pooled_output1,pooled_output2),1)
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        ########################多层感知机分类
        # pooled_output = torch.cat((pooled_output1,pooled_output2),1)
        # pooled_output = self.dropout(pooled_output)
        # logits = torch.relu(self.mlp1(pooled_output))
        # # logits = torch.relu(self.mlp2(logits))
        # logits = self.mlp2(logits)
        #############3层感知机
        pooled_output = torch.cat((pooled_output1,pooled_output2),1)
        pooled_output = self.dropout(pooled_output)
        logits = torch.relu(self.mlp1(pooled_output))
        logits = torch.relu(self.mlp2(logits))
        logits = self.mlp3(logits)
        ######### 4层感知机
        # pooled_output = torch.cat((pooled_output1,pooled_output2),1)
        # pooled_output = self.dropout(pooled_output)
        # logits = torch.relu(self.mlp1(pooled_output))
        # logits = torch.relu(self.mlp2(logits))
        # logits = torch.relu(self.mlp3(logits))
        # logits = self.mlp4(logits)
        ##############问句与答案字符串的编码和语义相似度编码不采用同一个bert,分别映射到2维再相加##############
        # import pdb; pdb.set_trace()
        # logits1 = self.classifier1(self.dropout(pooled_output1))
        # logits2 = self.classifier2(self.dropout2(pooled_output2))
        # logits = (logits1 + logits2)
        # logits = logits1 * torch.sigmoid(logits2)
        # logits = logits1 + logits1 * torch.sigmoid(logits2)
        # logits = logits1 * torch.sigmoid(logits2)
        # logits = (logits1 * torch.tanh(logits2) + logits1) / 2.0
        # logits = logits1 * torch.tanh(logits2)
        # logits = logits1 + torch.tanh(logits1) * logits2
        # logits = logits1 + torch.max(logits1, logits2)
        # logits = logits1 + torch.max(logits1, logits1 * torch.tanh(logits2))
        # logits = torch.tanh(logits1) * logits2
        # logits = logits1
        return logits

    # def __init__(self, config, num_labels):
    #     super(BertFor2PairSequenceWithAnswerTypeMidDim, self).__init__(config)
    #     self.num_labels = num_labels
    #     self.bert = BertModel(config)
    #     self.dropout = nn.Dropout(config.hidden_dropout_prob)
    #     self.classifier = nn.Linear(config.hidden_size, num_labels)
    #     self.apply(self.init_bert_weights)

    # def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
    #     input_ids = input_ids.view(-1, 2, 100)
    #     token_type_ids = token_type_ids.view(-1, 2, 100)
    #     attention_mask = attention_mask.view(-1, 2, 100)
    #     input_ids1 = input_ids[:, 0, :]
    #     input_ids2 = input_ids[:, 1, :]
    #     token_type_ids1 = token_type_ids[:, 0, :]
    #     token_type_ids2 = token_type_ids[:, 1, :]
    #     attention_mask1 = attention_mask[:, 0, :]
    #     attention_mask2 = attention_mask[:, 1, :]
    #     _, pooled_output = self.bert(input_ids1, token_type_ids1, attention_mask1, output_all_encoded_layers=False)
    #     # import pdb; pdb.set_trace()
    #     # _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
    #     # num_sen = pooled_output.shape[0]
    #     pooled_output = self.dropout(pooled_output)
    #     logits = self.classifier(pooled_output)
    #     # print(logits)
    #     return logits


class BertFor3PairSequenceWithAnswer(BertPreTrainedModel):
    
    def __init__(self, config, num_labels):
        super(BertFor3PairSequenceWithAnswer, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.bert2 = BertModel(config)
        # self.bert3 = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 3, num_labels)
        self.classifier1 = nn.Linear(config.hidden_size, num_labels)
        self.classifier2 = nn.Linear(config.hidden_size, num_labels)
        self.classifier3 = nn.Linear(config.hidden_size, num_labels)
        # self.denseCat = nn.Linear(config.hidden_size * 3, config.hidden_size)
        # self.activation = nn.Tanh()
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, rels_ids = None):
        ##############问句与答案字符串的编码和语义相似度编码不采用同一个bert##############
        input_ids = input_ids.view(-1, 3, 100)
        token_type_ids = token_type_ids.view(-1, 3, 100)
        attention_mask = attention_mask.view(-1, 3, 100)
        input_ids1 = input_ids[:, 0, :]
        input_ids2 = input_ids[:, 1, :]
        input_ids3 = input_ids[:, 2, :]
        token_type_ids1 = token_type_ids[:, 0, :]
        token_type_ids2 = token_type_ids[:, 1, :]
        token_type_ids3 = token_type_ids[:, 2, :]
        attention_mask1 = attention_mask[:, 0, :]
        attention_mask2 = attention_mask[:, 1, :]
        attention_mask3 = attention_mask[:, 2, :]
       
        # pooled_output = torch.cat((pooled_output1,pooled_output2, pooled_output3),1)
        # denseCat = self.denseCat(pooled_output)
        # denseCat = self.activation(denseCat)
        # pooled_output = self.dropout(denseCat)
        ###############拼接之后直接分类#####################3
        # logits = self.classifier(self.dropout(pooled_output))
        ###########直接映射到2维进行相加###########################
        # _, pooled_output1 = self.bert(input_ids1, token_type_ids1, attention_mask1)
        # _, pooled_output2 = self.bert2(input_ids2, token_type_ids2, attention_mask2)
        # _, pooled_output3 = self.bert3(input_ids3, token_type_ids3, attention_mask3)
        # pooled_output1 = self.dropout(pooled_output1)
        # pooled_output2 = self.dropout(pooled_output2)
        # pooled_output3 = self.dropout(pooled_output3)
        # logits = self.classifier1(pooled_output1) + self.classifier2(pooled_output2) + self.classifier3(pooled_output3)
        #########答案信息共用一个bert################
        _, pooled_output1 = self.bert(input_ids1, token_type_ids1, attention_mask1)
        _, pooled_output2 = self.bert2(input_ids2, token_type_ids2, attention_mask2)
        _, pooled_output3 = self.bert2(input_ids3, token_type_ids3, attention_mask3)
        pooled_output1 = self.dropout(pooled_output1)
        pooled_output2 = self.dropout(pooled_output2)
        pooled_output3 = self.dropout(pooled_output3)
        logits = 0.7 * self.classifier1(pooled_output1) + 0.2 * self.classifier2(pooled_output2) + 0.1 * self.classifier3(pooled_output3)
        # import pdb; pdb.set_trace()
        return logits



# class BertForSequenceClassification_new_sub2(BertPreTrainedModel):
#     def __init__(self, config, num_labels):
#         super(BertForSequenceClassification_new_sub2, self).__init__(config)
#         self.num_labels = num_labels
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, num_labels)
#         self.classifier_2 = nn.Linear(config.hidden_size, num_labels)
#         self.apply(self.init_bert_weights)

#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
#         _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
#         num_sen = pooled_output.shape[0]
#         cat_torch = torch.rand((num_sen // 2, 768), device='cuda:0')
#         try:
#             for i in range(0, num_sen, 2):
#                 cat_torch[i // 2] = pooled_output[i] - pooled_output[i + 1]
#         except:
#             import pdb; pdb.set_trace()
#         # import pdb; pdb.set_trace()
#         pooled_output = self.dropout(cat_torch)
#         logits = self.classifier_2(pooled_output).view(num_sen // 2, -1)
#         # logits = torch.max(logits_1, 1)[0]
#         # logits = torch.mean(logits_1, 1)
#         # logits = torch.min(logits_1, 1)[0]
#         # import pdb; pdb.set_trace()
#         # import pdb; pdb.set_trace()
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             return loss
#         else:
#             return logits


# class BertForSequenceClassification_new_sub(BertPreTrainedModel):
#     def __init__(self, config, num_labels):
#         super(BertForSequenceClassification_new_sub, self).__init__(config)
#         self.num_labels = num_labels
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, num_labels)
#         self.classifier_2 = nn.Linear(config.hidden_size, num_labels)
#         self.apply(self.init_bert_weights)

#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
#         _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
#         num_sen = pooled_output.shape[0]
#         cat_torch = torch.rand((num_sen // 2, 768), device='cuda:0')
#         try:
#             for i in range(0, num_sen, 2):
#                 cat_torch[i // 2] = pooled_output[i] - pooled_output[i + 1]
#         except:
#             import pdb; pdb.set_trace()
#         # import pdb; pdb.set_trace()
#         pooled_output = self.dropout(cat_torch)
#         logits = self.classifier_2(pooled_output).view(num_sen // 2, -1)
#         # logits = torch.max(logits_1, 1)[0]
#         # logits = torch.mean(logits_1, 1)
#         # logits = torch.min(logits_1, 1)[0]
#         # import pdb; pdb.set_trace()
#         # import pdb; pdb.set_trace()
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             return loss
#         else:
#             return logits


# class BertForSequenceClassification_new(BertPreTrainedModel):
#     """BERT model for classification.
#     This module is composed of the BERT model with a linear layer on top of
#     the pooled output.

#     Params:
#         `config`: a BertConfig class instance with the configuration to build a new model.
#         `num_labels`: the number of classes for the classifier. Default = 2.

#     Inputs:
#         `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
#             with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
#             `extract_features.py`, `run_classifier.py` and `run_squad.py`)
#         `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
#             types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
#             a `sentence B` token (see BERT paper for more details).
#         `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
#             selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
#             input sequence length in the current batch. It's the mask that we typically use for attention when
#             a batch has varying length sentences.
#         `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
#             with indices selected in [0, ..., num_labels].

#     Outputs:
#         if `labels` is not `None`:
#             Outputs the CrossEntropy classification loss of the output with the labels.
#         if `labels` is `None`:
#             Outputs the classification logits of shape [batch_size, num_labels].

#     Example usage:
#     ```python
#     # Already been converted into WordPiece token ids
#     input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
#     input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
#     token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

#     config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
#         num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

#     num_labels = 2

#     model = BertForSequenceClassification(config, num_labels)
#     logits = model(input_ids, token_type_ids, input_mask)
#     ```
#     """
#     def __init__(self, config, num_labels):
#         super(BertForSequenceClassification_new, self).__init__(config)
#         self.num_labels = num_labels
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, num_labels)
#         self.classifier_2 = nn.Linear(config.hidden_size * 2, num_labels)
#         self.apply(self.init_bert_weights)

#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
#         _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
#         num_sen = pooled_output.shape[0]
#         cat_torch = torch.rand((num_sen // 2, 768 * 2), device='cuda:0')
#         try:
#             for i in range(0, num_sen, 2):
#                 cat_torch[i // 2] = torch.cat((pooled_output[i], pooled_output[i + 1]), 0)
#         except:
#             import pdb; pdb.set_trace()
#         # import pdb; pdb.set_trace()
#         pooled_output = self.dropout(cat_torch)
#         logits = self.classifier_2(pooled_output).view(num_sen // 2, -1)
#         # logits = torch.max(logits_1, 1)[0]
#         # logits = torch.mean(logits_1, 1)
#         # logits = torch.min(logits_1, 1)[0]
#         # import pdb; pdb.set_trace()
#         # import pdb; pdb.set_trace()
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             return loss
#         else:
#             return logits


# class BertForSequenceClassification_split3(BertPreTrainedModel):
#     """BERT model for classification.
#     This module is composed of the BERT model with a linear layer on top of
#     the pooled output.

#     Params:
#         `config`: a BertConfig class instance with the configuration to build a new model.
#         `num_labels`: the number of classes for the classifier. Default = 2.

#     Inputs:
#         `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
#             with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
#             `extract_features.py`, `run_classifier.py` and `run_squad.py`)
#         `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
#             types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
#             a `sentence B` token (see BERT paper for more details).
#         `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
#             selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
#             input sequence length in the current batch. It's the mask that we typically use for attention when
#             a batch has varying length sentences.
#         `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
#             with indices selected in [0, ..., num_labels].

#     Outputs:
#         if `labels` is not `None`:
#             Outputs the CrossEntropy classification loss of the output with the labels.
#         if `labels` is `None`:
#             Outputs the classification logits of shape [batch_size, num_labels].

#     Example usage:
#     ```python
#     # Already been converted into WordPiece token ids
#     input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
#     input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
#     token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

#     config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
#         num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

#     num_labels = 2

#     model = BertForSequenceClassification(config, num_labels)
#     logits = model(input_ids, token_type_ids, input_mask)
#     ```
#     """
#     def __init__(self, config, num_labels):
#         super(BertForSequenceClassification_split3, self).__init__(config)
#         self.num_labels = num_labels
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, num_labels)
#         self.classifier_2 = nn.Linear(config.hidden_size * 2, num_labels)
#         self.apply(self.init_bert_weights)

#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
#         words_embedding, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
#         # 使用最大长度进行平均池化
#         # words_embedding = words_embedding.permute(0,2,1)
#         # seq_repr = torch.nn.functional.avg_pool1d(words_embedding, kernel_size=words_embedding.shape[-1]).squeeze(-1)
#         # ****************************************
#         # 根据实际长度进行平均池化
#         # seq_repr = torch.rand((words_embedding.shape[0], 768), device='cuda:0')
#         # for i, item in enumerate(attention_mask):
#         #     len_item = torch.sum(item)# 判断实际长度
#         #     seq_repr[i] = torch.mean(words_embedding[i][0:len_item], 0)
#             # import pdb; pdb.set_trace()
#         # *************************
#         # ************根据第一个‘CLS’和最后一个‘SEP’拼接作为特征
#         seq_repr = torch.rand((words_embedding.shape[0], 768 * 2), device='cuda:0')
#         for i, item in enumerate(attention_mask):
#             len_item = torch.sum(item)
#             seq_repr[i] = torch.cat((words_embedding[i][0], words_embedding[i][len_item-1]), 0)
#             # import pdb; pdb.set_trace()
#         pooled_output = self.dropout(seq_repr)
#         logits = self.classifier_2(pooled_output)
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             return loss
#         else:
#             return logits


# class BertForSequenceClassification_diff(BertPreTrainedModel):
#     """BERT model for classification.
#     This module is composed of the BERT model with a linear layer on top of
#     the pooled output.

#     Params:
#         `config`: a BertConfig class instance with the configuration to build a new model.
#         `num_labels`: the number of classes for the classifier. Default = 2.

#     Inputs:
#         `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
#             with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
#             `extract_features.py`, `run_classifier.py` and `run_squad.py`)
#         `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
#             types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
#             a `sentence B` token (see BERT paper for more details).
#         `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
#             selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
#             input sequence length in the current batch. It's the mask that we typically use for attention when
#             a batch has varying length sentences.
#         `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
#             with indices selected in [0, ..., num_labels].

#     Outputs:
#         if `labels` is not `None`:
#             Outputs the CrossEntropy classification loss of the output with the labels.
#         if `labels` is `None`:
#             Outputs the classification logits of shape [batch_size, num_labels].

#     Example usage:
#     ```python
#     # Already been converted into WordPiece token ids
#     input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
#     input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
#     token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

#     config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
#         num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

#     num_labels = 2

#     model = BertForSequenceClassification(config, num_labels)
#     logits = model(input_ids, token_type_ids, input_mask)
#     ```
#     """
#     def __init__(self, config, num_labels):
#         super(BertForSequenceClassification_diff, self).__init__(config)
#         self.num_labels = num_labels
#         self.bert = BertModel(config)
#         self.bert_1 = BertModel(config)
#         self.bert_2 = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size * 3, num_labels)
#         self.apply(self.init_bert_weights)

#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
#         input_ids_0 = input_ids[:, 0:100]
#         input_ids_1 = input_ids[:, 100:200]
#         input_ids_2 = input_ids[:, 200:]
#         token_type_ids_0 = token_type_ids[:, 0:100]
#         token_type_ids_1 = token_type_ids[:, 100:200]
#         token_type_ids_2 = token_type_ids[:, 200:]
#         attention_mask_0 = attention_mask[:, 0:100]
#         attention_mask_1 = attention_mask[:, 100:200]
#         attention_mask_2 = attention_mask[:, 200:]
        
#         _, pooled_output_0 = self.bert(input_ids_0, token_type_ids_0, attention_mask_0, output_all_encoded_layers=False)
#         _, pooled_output_1 = self.bert_1(input_ids_1, token_type_ids_1, attention_mask_1, output_all_encoded_layers=False)
#         _, pooled_output_2 = self.bert_2(input_ids_2, token_type_ids_2, attention_mask_2, output_all_encoded_layers=False)
#         # _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
#         # 三个子路径拼接再分类
#         pooled_output = torch.cat((pooled_output_0, pooled_output_1, pooled_output_2), 1)
#         # ***********************
#         import pdb; pdb.set_trace()
#         # 三个子路径进行平均池化
#         # pooled_output_all = torch.cat((pooled_output_0.view(-1, 1, 768), pooled_output_1.view(-1, 1, 768), pooled_output_2.view(-1, 1, 768)), 1)
#         # pooled_output_all = pooled_output_all.permute(0, 2, 1)
#         # pooled_output = torch.nn.functional.avg_pool1d(pooled_output_all, kernel_size=pooled_output_all.shape[-1]).squeeze(-1)
#         # ******************************


#         # pooled_output = torch.max(pooled_output_all, 1)[0]

#         # pooled_output = pooled_output_0
#         # import pdb; pdb.set_trace()
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)

#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             return loss
#         else:
#             return logits