# -*- coding:utf-8 -*-

import os
import time
import json
import random
import codecs
import pickle as cPickle
from parser import CoreNLPParser

from LogUtil import LogInfo
# from stanfordcorenlp import StanfordCoreNLP
# nlp = StanfordCoreNLP(r'/data2/yhjia/enviroment/stanford-corenlp-full-2018-02-27')

def pos_sentence(sent):
    parse = parser.parse(sent)
    pos_list = [t.pos for t in parse.tokens]
    return ' '.join(pos_list)


def weighted_sampling(item_list, weight_list, budget):
    if len(item_list) == 0:
        return []

    data_size = len(item_list)
    acc_list = list(weight_list)
    for i in range(1, data_size):
        acc_list[i] += acc_list[i-1]
    norm = acc_list[-1]
    acc_list = map(lambda score: score / norm, acc_list)

    sample_list = []
    for _ in range(budget):
        x = random.random()
        pick_idx = -1
        for i in range(data_size):
            if acc_list[i] >= x:
                pick_idx = i
                break
        sample_list.append(item_list[pick_idx])

    return sample_list


def load_compq():
    compq_path = '../../qa-corpus/MulCQA'
    pickle_fp = compq_path + '/compQ.all.cPickle'
    if os.path.isfile(pickle_fp):
        LogInfo.logs('Loading ComplexQuestions from cPickle ...')
        with open(pickle_fp, 'rb') as br:
            qa_list = cPickle.load(br)
    else:
        LogInfo.logs('CompQ initializing ... ')
        qa_list = []
        for Tvt in ('train', 'test'):
            fp = '%s/compQ.%s.release' % (compq_path, Tvt)
            with codecs.open(fp, 'r', 'utf-8') as br:
                for line in br.readlines():
                    qa = {}
                    q, a_list_str = line.strip().split('\t')
                    qa['utterance'] = q
                    qa['targetValue'] = json.loads(a_list_str)
                    # import pdb; pdb.set_trace()
                    # qa['tokens'] = nlp.word_tokenize(q)
                    qa_list.append(qa)
        # import pdb; pdb.set_trace()
        with open(pickle_fp, 'wb') as bw:
            cPickle.dump(qa_list, bw)
    LogInfo.logs('%d ComplexQuestions loaded.', len(qa_list))
    return qa_list


def read_concept_edge(amr_concept, amr_edge1, amr_edge2, amr_edge3, amr_edge4, amr_edge5):
    concept = amr_concept.split()
    edge1 = amr_edge1.split()
    edge2 = amr_edge2.split()
    edge3 = amr_edge3.split()
    edge4 = amr_edge4.split()
    edge5 = amr_edge5.split()
    edge = [[[] for j in range(len(concept))] for i in range(len(concept))]
    edge_mat = []
    for e1, e2, e3, e4, e5 in zip(edge1, edge2, edge3, edge4, edge5):
        edge_mat.append([e1, e2, e3, e4, e5])
    if not len(edge1) == len(edge2) == len(edge3) == len(edge4) == len(edge5) == len(concept) ** 2:
        __import__('pdb').set_trace()
    for i in range(len(concept)):
        for j in range(len(concept)):
            edge[i][j] = edge_mat[i*len(concept)+j]
    return concept, edge

def load_webq():
    webq_path = '../../qa-corpus/web-question'
    pickle_fp = webq_path + '/webquestions.all.cPickle'
    if os.path.isfile(pickle_fp):
        LogInfo.logs('Loading Webquestions from cPickle ...')
        with open(pickle_fp, 'rb') as br:
            qa_list = cPickle.load(br)
    else:
        LogInfo.logs('WebQ initializing ... ')
        qa_list = []
        for Tvt in ('train', 'test'):
            webq_fp = f'{webq_path}/data/webquestions.examples.{Tvt}.json'
            with codecs.open(webq_fp, 'r', 'utf-8') as br:
                webq_data = json.load(br)
            # import pdb; pdb.set_trace()
            for raw_info in webq_data:
                qa = {}
                target_value = []
                ans_line = raw_info['targetValue']
                ans_line = ans_line[7: -2]      # remove '(list (' and '))'
                for ans_item in ans_line.split(') ('):
                    ans_item = ans_item[12:]    # remove 'description '
                    if ans_item.startswith('"') and ans_item.endswith('"'):
                        ans_item = ans_item[1: -1]
                    target_value.append(ans_item)
                qa['utterance'] = raw_info['utterance']
                qa['targetValue'] = target_value
                # import pdb; pdb.set_trace()
                # qa['tokens'] = nlp.word_tokenize(raw_info['utterance'])
                qa_list.append(qa)
                if len(qa_list) % 1000 == 0:
                    LogInfo.logs('%d / %d scanned.', len(qa_list), len(webq_data))
        # import pdb; pdb.set_trace()
        # qa_list中每个元素格式:{'utterance': 'what is the name of justin bieber brother?', 'targetValue': ['Jazmyn Bieber', 'Jaxon Bieber']}
        with open(pickle_fp, 'wb') as bw:
            cPickle.dump(qa_list, bw)
    LogInfo.logs('%d WebQuesetions loaded.', len(qa_list))
    return qa_list


def load_simpq():
    simpq_path = '/data2/jzzhou/research-code/acl-1209-amr-kbqa/aaai20-qa-junhui/qa-corpus/simple-question'
    pickle_fp = simpq_path + '/simpQ.all.cPickle'
    st = time.time()
    if os.path.isfile(pickle_fp):
        LogInfo.logs('Loading SimpleQuestions from cPickle ...')
        with open(pickle_fp, 'rb') as br:
            qa_list = cPickle.load(br)
    else:
        LogInfo.logs('SimpQ initializing ... ')
        qa_list = []
        for Tvt in ('train', 'valid', 'test'):
            fp = '%s/annotated_fb_data_%s.txt' % (simpq_path, Tvt)
            with codecs.open(fp, 'r', 'utf-8') as br:
                for line in br.readlines():
                    qa = {}
                    s, p, o, q = line.strip().split('\t')
                    s = remove_simpq_header(s)
                    p = remove_simpq_header(p)
                    o = remove_simpq_header(o)
                    qa['utterance'] = q
                    qa['targetValue'] = (s, p, o)       # different from other datasets
                    # qa['tokens'] = parser.parse(qa['utterance']).tokens
                    qa['parse'] = parser.parse(qa['utterance'])
                    qa['tokens'] = qa['parse'].tokens
                    qa_list.append(qa)
                    if len(qa_list) % 1000 == 0:
                        LogInfo.logs('%d scanned.', len(qa_list))
        with open(pickle_fp, 'wb') as bw:
            cPickle.dump(qa_list, bw)
    LogInfo.logs('[%.3fs] %d SimpleQuestions loaded.', time.time() - st, len(qa_list))
    return qa_list


def load_lcquadv1():
    fp = '/data2/pjzhang/aaai20-qa-junhui/qa-corpus/LC-QuAD'
    pickle_fp = fp + '/lcquad.all.cPickle'
    if os.path.isfile(pickle_fp):
        LogInfo.logs('Loading lcquad-v1.0 from cPickle ...')
        with open(pickle_fp, 'rb') as br:
            qa_list = cPickle.load(br)
    else:
        LogInfo.logs('LC-QuADv1.0 initializing ... ')
        qa_list = []
        # amr_fp = f'{fp}/amr/web.amr.pre'
        # amr_res = open(amr_fp, 'r').readlines()
        for Tvt in ('train', 'test'):
            lcquad_fp = f'{fp}/{Tvt}-data.json'
            with codecs.open(lcquad_fp, 'r', 'utf-8') as br:
                qa_data = json.load(br)
            for raw_info in qa_data:
                qa = {}
                qa['utterance'] = raw_info['corrected_question']
                #  qa['targetValue'] = target_value
                qa['parse'] = parser.parse(qa['utterance'])
                qa['tokens'] = qa['parse'].tokens
                qa['id'] = raw_info['_id']
                #  qa['amr'] = amr_res[len(qa_list)].strip()

                qa_list.append(qa)
                if len(qa_list) % 1000 == 0:
                    LogInfo.logs('%d / %d scanned.', len(qa_list), len(qa_data))
        with open(pickle_fp, 'wb') as bw:
            cPickle.dump(qa_list, bw)
    LogInfo.logs('%d LCQuADv1.0 loaded.', len(qa_list))
    return qa_list


def load_lcquadv2():
    fp = '/data2/pjzhang/aaai20-qa-junhui/qa-corpus/LC-QuAD2'
    pickle_fp = fp + '/lcquad.all.cPickle'
    if os.path.isfile(pickle_fp):
        LogInfo.logs('Loading lcquad-v2.0 from cPickle ...')
        with open(pickle_fp, 'rb') as br:
            qa_list = cPickle.load(br)
    else:
        LogInfo.logs('LC-QuADv2.0 initializing ... ')
        qa_list = []
        # amr_fp = f'{fp}/amr/web.amr.pre'
        # amr_res = open(amr_fp, 'r').readlines()
        for Tvt in ('train', 'test'):
            lcquad_fp = f'{fp}/{Tvt}.json'
            with codecs.open(lcquad_fp, 'r', 'utf-8') as br:
                qa_data = json.load(br)
            for raw_info in qa_data:
                if raw_info['question'] is None:
                    continue
                if len(raw_info['question']) == 0:
                    continue
                qa = {}
                qa['utterance'] = raw_info['question']
                #  qa['targetValue'] = target_value
                qa['parse'] = parser.parse(qa['utterance'])
                qa['tokens'] = qa['parse'].tokens
                qa['id'] = raw_info['uid']
                #  qa['amr'] = amr_res[len(qa_list)].strip()
                qa_list.append(qa)

                qa['utterance'] = raw_info['paraphrased_question']
                qa['parse'] = parser.parse(qa['utterance'])
                qa['tokens'] = qa['parse'].tokens
                qa['id'] = raw_info['uid']
                qa_list.append(qa)
                if len(qa_list) % 1000 == 0:
                    LogInfo.logs('%d / %d scanned.', len(qa_list), len(qa_data) * 2)
        with open(pickle_fp, 'wb') as bw:
            cPickle.dump(qa_list, bw)
    LogInfo.logs('%d LCQuADv2.0 loaded.', len(qa_list))
    return qa_list


def remove_simpq_header(mid):
    mid = mid[17:]  # remove www.freebase.com/
    mid = mid.replace('/', '.')
    return mid


def load_simpq_sp_dict(fb='FB2M'):
    sp_dict_fp = f'/data2/jzzhou/research-code/kbqa-corpus/knowledge-graph/freebase-subsets/freebase-{fb}.txt'
    sp_dict = {}
    with codecs.open(sp_dict_fp, 'r', 'utf-8') as br:
        for line in br.readlines():
            spt = line.strip().split('\t')
            subj = spt[0]
            pred_set = set(spt[1:])
            sp_dict[subj] = pred_set
    LogInfo.logs('%d <entity, pred_set> loaded from %s.', len(sp_dict), fb)
    return sp_dict


def get_q_range_by_mode(data_name, mode):
    assert mode in ('train', 'valid', 'test')
    assert data_name in ('SimpQ', 'WebQ', 'CompQ')

    q_idx_list = []
    if data_name == 'WebQ':             # 3023 / 3778 / 2032
        if mode == 'train':
            q_idx_list = range(3023)
        elif mode == 'valid':
            q_idx_list = range(3023, 3778)
        elif mode == 'test':
            q_idx_list = range(3778, 1000000)
    elif data_name == 'SimpQ':          # 75910 / 86755 / 108442
        if mode == 'train':
            q_idx_list = range(75910)
        elif mode == 'valid':
            q_idx_list = range(75910, 86755)
        elif mode == 'test':
            q_idx_list = range(86755, 1000000)
    elif data_name == 'CompQ':
        if mode == 'train':
            q_idx_list = range(1000)
        elif mode == 'valid':
            q_idx_list = range(1000, 1300)
        elif mode == 'test':
            q_idx_list = range(1300, 2100)
    return q_idx_list


def main():
    qa_list = load_simpq()
    for idx in range(4):
        qa = qa_list[idx]
        LogInfo.logs('utterance: %s', qa['utterance'].encode('utf-8'))
        LogInfo.logs('tokens: %s', [tok.token for tok in qa['tokens']])


if __name__ == '__main__':
    main()
