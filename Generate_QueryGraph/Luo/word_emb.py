# -*- coding: utf-8 -*-

# import re
# import os
import numpy as np
import pickle as cPickle

# from ..linking import parser

from LogUtil import LogInfo


class WordEmbeddingUtil(object):

    def __init__(self, wd_emb, dim_emb,
                 emb_dir='../../data/word_emb_in_use',
                 # parser_ip='202.120.38.146',
                 # parser_port=9601):     # BH: 9601; DS: 8601
                 ):                
        self.word_dict_fp = '../../data/word_emb_in_use/word_emb.indices' 
        self.word_emb_mat_fp = '../../data/word_emb_in_use/word_emb.glove_300.npy'
        self.dim_emb = dim_emb
        # {'maize': 83018, 'ypsilanti': 46353, 'conley': 51860···}
        self.word_idx_dict = None
        self.word_idx_token_dict = {}
        self.word_emb_matrix = None
        self.n_words = None

        # {'m.0b3w0dc': 141426, 'base.infrastructure.landfill.waste_treatment_methods': 105186···}
        self.mid_idx_dict = None  
        self.mid_emb_matrix = None
        self.n_mids = None
        # self.trans_word2id = './data/side_info/relation2id.pkl'
        # self.trans_id2emb = './data/side_info/relation2vec.pkl'

        # self.types_pretrain_file = './pretrain/word2vec_300d.txt'

        self.load_word_indices() # 返回self.word_idx_dict <word, idx> 
        self.load_word_embeddings() #返回self.word_emb_matrix 词向量
        # self.load_mid_indices() # 返回self.mid_idx_dict <mid, idx>
        # self.load_mid_embeddings() # 返回self.mid_emb_matrix mid向量
        self.dep_name_dict = {}

        
        with open(emb_dir+'/dep_names.txt', 'r') as br:
            for line in br.readlines():
                dep, name = line.strip().split('\t')
                self.dep_name_dict[dep] = name
        LogInfo.logs('%d dependency name loaded.', len(self.dep_name_dict))
        # import pdb; pdb.set_trace()

    def produce_active_dep_embedding(self):
        if self.dep_emb_fp is None:
            loginfo.logs("loading dependency error")
        else:
            vocab = {}
            vector = []
            i = 1
            fr = open(self.dep_emb_fp, "r")  
            line = fr.readline().strip()
            word_dim = int(line.split(" ")[1])
            vocab['unk'] = 0
            vector.append([0]*word_dim)
            for line in fr :
                row = line.strip().split(" ")
                vocab[row[0]] = i
                vector.append(row[1:])
                i+=1
            fr.close()
            return vocab,vector

    def produce_active_dep_3_embedding(self):
        if self.dep_emb_3_fp is None:
            loginfo.logs("loading dependency error")
        else:
            vocab = {}
            vector = []
            i = 1
            fr = open(self.dep_emb_3_fp, "r")  
            line = fr.readline().strip()
            word_dim = int(line.split(" ")[1])
            vocab['unk'] = 0
            vector.append([0]*word_dim)
            for line in fr :
                row = line.strip().split(" ")
                vocab[row[0]] = i
                vector.append(row[1:])
                i+=1
            fr.close()
            return vocab,vector

    def produce_pseq_trans_embedding(self, pseq_dic):
        trans_emb_matrix = np.random.uniform(low=-0.1, high=0.1,size=(len(pseq_dic), 50)).astype('float32') # 定义一个完整的随机向量矩阵
        f_id = open(self.trans_word2id, 'rb')
        f_emb = open(self.trans_id2emb, 'rb')
        relation2id = cPickle.load(f_id)    # relation2id是字典
        relation2vec = cPickle.load(f_emb)  #relation2vec是numpy矩阵
        mask_percentage = 0.0
        num = 0
        mask_num = 0
        for pseq in pseq_dic:
            num += 1
            if(pseq in relation2id):
                if(num * mask_percentage < mask_num):
                # import pdb; pdb.set_trace()
                    trans_emb_matrix[pseq_dic[pseq]] = relation2vec[int(relation2id[pseq])]  #在对应位置上将随机化向量替换为transE向量
                else:
                    mask_num += 1
                    # print('mask_num:', mask_num)
                # import pdb; pdb.set_trace()
            else:
                mask_num += 1
        print('实际上mask掉的关系词：', mask_num, num, mask_num * 1.0 / num)
        # import pdb; pdb.set_trace()
        return trans_emb_matrix

    # def produce_entity_types_embedding(self):
    #     f = open(self.types_pretrain_file, 'r', encoding='utf-8')
    #     type_dicts = {}
    #     type_dicts['<PAD>'] = 0
    #     type_dicts['<START>'] = 1
    #     type_dicts['<UNK>'] = 2
    #     lines = f.readlines()
    #     for line in lines:
    #         line_cut = line.strip().split(' ')
    #         if(len(line_cut) == 301):
    #             if(line_cut[0] not in type_dicts):
    #                 type_dicts[line_cut[0]] = len(type_dicts)
    #     types_emb_matrix = np.random.uniform(low=-0.1, high=0.1,size=(len(type_dicts), 300)).astype('float32') # 定义一个完整的随机向量矩阵
    #     for line in lines:
    #         line_cut = line.strip().split(' ')
    #         if(len(line_cut) == 301):
    #             type_id = type_dicts[line_cut[0]]
    #             i = 1
    #             while i < len(line_cut):
    #                 types_emb_matrix[type_id,i-1] = float(line_cut[i])
    #                 i += 1
    #     import pdb; pdb.set_trace()
    #     return type_dicts, types_emb_matrix

    def load_word_idx_indices(self):
        if self.word_idx_dict is None:
            LogInfo.logs('Loading <word, idx> pairs from [%s] ... ', self.word_dict_fp)
            with open(self.word_dict_fp, 'rb') as br:
                self.word_idx_dict = cPickle.load(br)   # 加载词向量_id
            LogInfo.logs('%d <word, idx> loaded.', len(self.word_idx_dict))           
            for i,j in self.word_idx_dict.items():
                self.word_idx_token_dict[j] = i
            self.n_words = len(self.word_idx_dict)
        return self.word_idx_token_dict

    def load_word_indices(self):
        if self.word_idx_dict is None:
            LogInfo.logs('Loading <word, idx> pairs from [%s] ... ', self.word_dict_fp)
            with open(self.word_dict_fp, 'rb') as br:
                self.word_idx_dict = cPickle.load(br)   # 加载词向量_id
            LogInfo.logs('%d <word, idx> loaded.', len(self.word_idx_dict))           
            for i,j in self.word_idx_dict.items():
                self.word_idx_token_dict[j] = i
            self.n_words = len(self.word_idx_dict)
        return self.word_idx_dict

    def load_word_embeddings(self):
        if self.word_emb_matrix is None:
            LogInfo.logs('Loading word embeddings for [%s] ...', self.word_emb_mat_fp)
            self.word_emb_matrix = np.load(self.word_emb_mat_fp)   # 加载词向量矩阵
            LogInfo.logs('%s word embedding loaded.', self.word_emb_matrix.shape)
            assert self.word_emb_matrix.shape == (self.n_words, self.dim_emb)
        return self.word_emb_matrix

    def load_mid_indices(self):
        if self.mid_idx_dict is None:
            LogInfo.logs('Loading <mid, idx> pairs from [%s] ... ', self.mid_dict_fp)
            with open(self.mid_dict_fp, 'rb') as br:
                self.mid_idx_dict = cPickle.load(br)        #  加载词_mid的id 
            LogInfo.logs('%d <mid, idx> loaded.', len(self.mid_idx_dict))
            self.n_mids = len(self.mid_idx_dict)
        return self.mid_idx_dict

    def load_mid_embeddings(self):
        if self.mid_emb_matrix is None:
            LogInfo.logs('Loading mid embeddings for [%s] ...', self.mid_emb_mat_fp)
            self.mid_emb_matrix = np.load(self.mid_emb_mat_fp)   #  加载词mid的词向量矩阵
            LogInfo.logs('%s mid embedding loaded.', self.mid_emb_matrix.shape)
            assert self.mid_emb_matrix.shape == (self.n_mids, self.dim_emb)
        return self.mid_emb_matrix

    def produce_active_word_embedding(self, active_word_dict, dep_simulate=False):
        active_size = len(active_word_dict)
        word_emb_matrix = np.random.uniform(low=-0.1, high=0.1,size=(active_size, self.dim_emb)).astype('float32')
        self.load_word_indices()
        self.load_word_embeddings()
        # f = open('./glove_words_id.txt', 'w', encoding='utf-8')
        # for item in self.word_idx_dict:
        #     f.write(item + '\t' + str(self.word_idx_dict[item]) + '\n')
        # f.flush()
        # import pdb; pdb.set_trace()
        num = 0
        for tok, active_idx in active_word_dict.items():
            # import pdb; pdb.set_trace()
            if tok in self.word_idx_dict:
                local_idx = self.word_idx_dict[tok]
                word_emb_matrix[active_idx] = self.word_emb_matrix[local_idx]
            # else:
            #     print(tok, num)
            #     num += 1
        # import pdb; pdb.set_trace()

        LogInfo.logs('%s active word embedding created.', word_emb_matrix.shape)
        return word_emb_matrix

    def produce_active_mid_embedding(self, active_mid_dict):
        active_size = len(active_mid_dict)
        mid_emb_matrix = np.random.uniform(low=-0.1, high=0.1,
                                           size=(active_size, self.dim_emb)).astype('float32')
        self.load_mid_indices()
        self.load_mid_embeddings()
        for tok, active_idx in active_mid_dict.items():
            if tok in self.mid_idx_dict:
                local_idx = self.mid_idx_dict[tok]
                mid_emb_matrix[active_idx] = self.mid_emb_matrix[local_idx]
        LogInfo.logs('%s active mid embedding created.', mid_emb_matrix.shape)
        return mid_emb_matrix

    def get_phrase_emb(self, phrase):
        """
        Calculate the embedding of a new phrase, by averaging the embeddings of all observed words
        """
        self.load_word_embeddings()
        if phrase == '':
            return None
        spt = phrase.split(' ')
        idx_list = []
        for wd in spt:
            if wd in self.word_idx_dict:
                idx_list.append(self.word_idx_dict[wd])
        if len(idx_list) == 0:
            return None
        emb = np.mean(self.word_emb_matrix[idx_list], axis=0)  # (n_words, dim_emb) ==> (dim_emb, )
        emb = emb / np.linalg.norm(emb)  # normalization
        return emb
