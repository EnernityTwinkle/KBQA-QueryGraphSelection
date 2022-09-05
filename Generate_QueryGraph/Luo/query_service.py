"""
Author: Kangqi Luo
Date: 180208
Goal: The service for handling all SPARQL queries & maintain all queried schemas.
"""

import re
import os
import sys
import json
import codecs
from datetime import datetime

"""python3 version"""
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler

from u import load_compq, load_webq
from official_eval import compute_f1

from backend import SPARQLHTTPBackend

from LogUtil import LogInfo


def show_time():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class QueryService:
    def __init__(self, sparql_cache_fp, qsc_cache_fp, vb=0):
        LogInfo.begin_track('QueryService initialize ... ')
        self.year_re = re.compile(r'^[1-2][0-9][0-9][0-9]$')   # 年份正则表达式
        self.query_prefix = 'PREFIX fb: <http://rdf.freebase.com/ns/> '
        self.pref_len = len(self.query_prefix)
        self.sparql_dict = {}       # key: sparql, value: query_ret
        self.q_sc_dict = {}         # key: q_idx + sparql + count_or_not, value: p, r, f1, ans_size
        self.q_gold_dict = {}       # key: q_id (WebQ-xxx, CompQ-xxx), value: gold answers (with possible preprocess)
        self.sparql_buffer = []
        self.q_sc_buffer = []
        # self.lock = lock        # control multiprocess
        self.backend = SPARQLHTTPBackend('192.168.126.124', '8999', '/sparql', cache_enabled=False)
        self.vb = vb
        # vb = 0: silence
        # vb = 1: show request information or cache hit information
        # vb = 2: show the full request information, no matter hit or not
        # vb = 3: show detail return value
        compq_list = load_compq() # 得到ComplexQ中的问句和对应的答案序列
        webq_list = load_webq() # 得到WebQ中的问句和对应的答案序列
        for mark, qa_list in [('CompQ', compq_list), ('WebQ', webq_list)]:
            for idx, qa in enumerate(qa_list):
                gold_list = qa['targetValue']
                q_id = '%s_%d' % (mark, idx)
                self.q_gold_dict[q_id] = gold_list # 按照问句的id将每个问句的答案存储在self.q_gold_dict中
                # self.q_gold_dict内容格式为：{'CompQ_0': ['delegate to the continental congress']}
        LogInfo.logs('%d QA loaded from WebQ & CompQ.', len(self.q_gold_dict))
        # import pdb; pdb.set_trace()
        self.sparql_cache_fp = sparql_cache_fp
        self.qsc_cache_fp = qsc_cache_fp
        if not os.path.isfile(self.sparql_cache_fp):
            os.makedirs('/'.join(self.sparql_cache_fp.split('/')[:-1]), exist_ok=True)
            os.mknod(self.sparql_cache_fp)
        if not os.path.isfile(self.qsc_cache_fp):
            os.makedirs('/'.join(self.qsc_cache_fp.split('/')[:-1]), exist_ok=True)
            os.mknod(self.qsc_cache_fp)
        LogInfo.begin_track('Loading SPARQL cache ...')
        if os.path.isfile(self.sparql_cache_fp):
            with codecs.open(self.sparql_cache_fp, 'r', 'utf-8') as br:
                while True:
                    line = br.readline()
                    if line is None or line == '':
                        break
                    key, query_ret = json.loads(line)
                    self.sparql_dict[key] = query_ret
        # import pdb; pdb.set_trace()
        LogInfo.logs('%d SPARQL cache loaded.', len(self.sparql_dict))
        LogInfo.end_track()
        LogInfo.begin_track('Loading <q_sc, stat> cache ...')
        # import pdb; pdb.set_trace()
        if os.path.isfile(self.qsc_cache_fp):
            with codecs.open(self.qsc_cache_fp, 'r', 'utf-8') as br:
                while True:
                    line = br.readline()
                    if line is None or line == '':
                        break
                    key, stat = json.loads(line)
                    self.q_sc_dict[key] = stat
        LogInfo.logs('%d <q_sc, stat> cache loaded.', len(self.q_sc_dict))
        LogInfo.end_track()
        LogInfo.end_track('Initialize complete.')
        # import pdb; pdb.set_trace()
    """ ====================== SPARQL related ====================== """

    def shrink_query(self, sparql_query):
        key = sparql_query
        if key.startswith(self.query_prefix):
            key = key[self.pref_len:]
        return key

    def kernel_query(self, key, repeat_count=10):
        sparql_str = self.query_prefix + key
        try_times = 0
        query_ret = None
        """ 180323: Try several times, if encountered exception """
        while try_times < repeat_count:
            try_times += 1
            query_ret = self.backend.query(sparql_str)
            if self.vb >= 1:
                if query_ret is not None:
                    LogInfo.logs('Query return lines = %d', len(query_ret))
                else:
                    LogInfo.logs('Query return None (exception encountered)'
                                 '[try_times=%d/%d]', try_times, repeat_count)
            if query_ret is not None:
                break
        return query_ret

    @staticmethod
    def extract_forbidden_mid(sparql_query):
        # extract all entities in the SPARQL query.
        forbidden_mid_set = set([])
        cur_st = 0
        while True:
            st = sparql_query.find('fb:m.', cur_st)
            if st == -1:
                break
            ed = sparql_query.find(' ', st + 5)
            mid = sparql_query[st+3: ed]
            forbidden_mid_set.add(mid)
            cur_st = ed
        return forbidden_mid_set

    def answer_post_process_new(self, query_ret, forbidden_mid_set, ret_symbol_list,
                                tm_comp, tm_value, allow_forever, ord_comp, ord_rank):
        """
        Deal with time filtering and ranking in this post-process.
        We can handle complex queries, such as time constraints on an interval, or ranking with ties.
        1. tm_comp: < / == / > / None
        2. tm_value: xxxx_yyyy / (blank)
        3. allow_forever: Fyes / Fdyn / Fhalf / Fno / (blank)
        4. ord_comp: min / max / None
        5. ord_rank: xxx / (blank)
        """
        """ Step 1: make sure what does each column represent """
        query_ret = list(filter(lambda _tup: len(_tup) == len(ret_symbol_list), query_ret))       # remove error lines
        ans_mid_pos = ans_name_pos = tm_begin_pos = tm_end_pos = ord_pos = -1
        for idx, symbol in enumerate(ret_symbol_list):
            if symbol in ('?o1', '?o2'):
                ans_mid_pos = idx
            elif symbol in ('?n1', '?n2'):
                ans_name_pos = idx
            elif symbol == '?tm1':
                tm_begin_pos = idx
            elif symbol == '?tm2':
                tm_end_pos = idx
            elif symbol == '?ord':
                ord_pos = idx
        assert ans_mid_pos != -1        # there must have ?o1 or ?o2

        """ Step 2: Filter by time """
        if tm_begin_pos == -1:      # no time constraint available
            tm_filter_ret = query_ret
        else:       # compare interval [a, b] and [c, d]
            close_open_interval = False
            if allow_forever.endswith('-co'):       # represent time interval in [begin, end) style
                close_open_interval = True
                allow_forever = allow_forever[:-3]
            assert allow_forever in ('Fyes', 'Fdyn', 'Fhalf', 'Fno')
            assert tm_comp in ('<', '==', '>')
            strict_ret = []     # both begin and end are given
            half_inf_ret = []   # one side is not given, treat as inf / -inf
            all_inf_ret = []    # both sides are not given, treat as unchangable fact
            target_begin_year, target_end_year = [int(x) for x in tm_value.split('_')]
            if tm_end_pos == -1:
                tm_end_pos = tm_begin_pos       # the schema returns a single time, not an interval
            for row in query_ret:
                begin_tm_str = row[tm_begin_pos]
                end_tm_str = row[tm_end_pos]
                begin_year = int(begin_tm_str[:4]) if re.match(self.year_re, begin_tm_str[:4]) else -12345
                end_year = int(end_tm_str[:4]) if re.match(self.year_re, end_tm_str[:4]) else 12345
                info_cate = 'strict'
                if begin_year == -12345 and end_year == 12345:
                    info_cate = 'all_inf'
                elif begin_year == -12345 or end_year == 12345:
                    info_cate = 'half_inf'

                # Taken "forever" into consideration, calculate the real state between target and output times
                if not close_open_interval:         # [begin, end]
                    if end_year < target_begin_year:
                        state = '<'
                    elif begin_year > target_end_year:
                        state = '>'
                    else:
                        state = '=='        # in, or say, during / overlap.
                else:                               # [begin, end) for both target and query result times
                    """ adjust the interval for the [begin, end) format """
                    if end_year <= begin_year:
                        end_year = begin_year + 1       # at least make sense for the first year
                    if target_end_year <= target_begin_year:
                        target_end_year = target_begin_year + 1
                    """ then compare [begin, end) with [target_begin, target_end) """
                    if end_year <= target_begin_year:
                        state = '<'
                    elif begin_year >= target_end_year:
                        state = '>'
                    else:
                        state = '=='
                if state == tm_comp:
                    if info_cate == 'strict':
                        strict_ret.append(row)
                    elif info_cate == 'half_inf':
                        half_inf_ret.append(row)
                    else:
                        all_inf_ret.append(row)
                # if('2000_2000' == tm_value):
                #     import pdb; pdb.set_trace()
            """ Merge strict / half_inf / all_inf rows based on different strategies """
            if allow_forever == 'Fno':      # ignore half_inf or all_inf
                tm_filter_ret = strict_ret
            elif allow_forever == 'Fhalf':
                tm_filter_ret = strict_ret + half_inf_ret
            elif allow_forever == 'Fyes':   # consider all cases
                tm_filter_ret = strict_ret + half_inf_ret + all_inf_ret
            else:       # Fdyn, consider all cases, but with priority
                if len(strict_ret) > 0:
                    tm_filter_ret = strict_ret
                elif len(half_inf_ret) > 0:
                    tm_filter_ret = half_inf_ret
                else:
                    tm_filter_ret = all_inf_ret
        if self.vb >= 3:
            LogInfo.logs('After time filter = %d', len(tm_filter_ret))
            LogInfo.logs(tm_filter_ret)
        """ Step 3: Ranking """
        if ord_pos == -1:
            ordinal_ret = tm_filter_ret             # no ordinal filtering available
        else:
            ordinal_ret = []
            ord_rank = int(ord_rank)
            assert ord_comp in ('min', 'max')
            if ord_rank <= len(tm_filter_ret):
                for row in tm_filter_ret:
                    ord_str = row[ord_pos]
                    """ Try convert string value into float, could be int/float/DT value """
                    val = None
                    try:                    # int/float/DT-year-only
                        val = float(ord_str)
                    except ValueError:      # datetime represented in YYYY-MM-DD style
                        hyphen_pos = ord_str.find('-', 1)   # ignore the beginning "-", which represents "negative".
                        if hyphen_pos != -1:        # remove MM-DD information
                            val = float(ord_str[:hyphen_pos])      # only picking year from datetime
                        else:       # still something wrong
                            LogInfo.logs('Warning: unexpected ordinal value "%s"', ord_str)
                    if val is not None:
                        row[ord_pos] = val
                        ordinal_ret.append(row)
                reverse = ord_comp == 'max'
                ordinal_ret.sort(key=lambda _tup: _tup[ord_pos], reverse=reverse)
                if self.vb >= 3:
                    LogInfo.logs('Sort by ordinal constraint ...')
                    LogInfo.logs(ordinal_ret)
                target_ord_value = ordinal_ret[ord_rank-1][ord_pos]
                LogInfo.logs('target_ord_value = %s', target_ord_value)
                ordinal_ret = list(filter(lambda _tup: _tup[ord_pos] == target_ord_value, ordinal_ret))
        if self.vb >= 3:
            LogInfo.logs('After ordinal filter = %d', len(ordinal_ret))
            LogInfo.logs(ordinal_ret)

        """ Step 4: Final answer collection """
        forbidden_ans_set = set([])     # all the names occurred in the query result whose mid is forbidden
        normal_ans_set = set([])        # all the remaining normal names
        for row in ordinal_ret:
            ans_mid = row[ans_mid_pos]  # could be mid/int/float/datetime
            if ans_name_pos == -1:          # the answer is int/float/datetime
                ans_name = ans_mid
                if re.match(self.year_re, ans_name[:4]):
                    ans_name = ans_name[:4]
                    # if we found a DT, then we just keep its year info.
                normal_ans_set.add(ans_name)
            else:
                ans_name = row[ans_name_pos]
                if ans_mid in forbidden_mid_set:
                    forbidden_ans_set.add(ans_name)
                else:
                    normal_ans_set.add(ans_name)

        if len(normal_ans_set) > 0:  # the normal answers have a strict higher priority.
            final_ans_set = normal_ans_set
        else:  # we take the forbidden answer as output, only if we have no other choice.
            final_ans_set = forbidden_ans_set
        LogInfo.logs('Final Answer: %s', final_ans_set)
        return final_ans_set

    @staticmethod
    def compq_answer_normalize(ans):
        """ Change hyphen and en-dash into '-', and then lower case. """
        return re.sub(u'[\u2013\u2212]', '-', ans).lower()

    """ ==================== Registered Functions ==================== """

    def query_sparql(self, sparql_query):
        '''
        功能：进行sparql语句搜索
        输入：sparql语句sparql_query
        输出：对应的搜索结果
        '''
        key = self.shrink_query(sparql_query)
        hit = key in self.sparql_dict  # 判断是否已经完成了sparql实体查询
        show_request = (not hit or self.vb >= 2)    # going to perform a real query, or just want to show more
        if show_request:
            LogInfo.begin_track('[%s] SPARQL Request:', show_time())
            LogInfo.logs(key)
        if hit and self.vb >= 1:
            if not show_request:
                LogInfo.logs('[%s] SPARQL hit!', show_time())
            else:
                LogInfo.logs('SPARQL hit!')
        if hit:                                    # 当前实体构建的sparql在实体链接构建的sparql中，直接将查找好的sparql语句中间结果---P所有谓词（查找结果）返回给query_ret
            query_ret = self.sparql_dict[key]
        else:                                      # 当前实体构建的sparql不在实体链接构建的sparql中，利用sparql语句去知识库中进行查询，得到最后查找结果。
            query_ret = self.kernel_query(key=key)
            if query_ret is not None:       # ignore schemas returning None
                self.sparql_dict[key] = query_ret
                self.sparql_buffer.append((key, query_ret))
        if show_request and self.vb >= 3:
            LogInfo.logs(query_ret)
        if show_request:
            LogInfo.end_track()
        final_query_ret = query_ret or []
        # import pdb; pdb.set_trace()
        return final_query_ret          # won't send None back, but use empty list as instead

    def query_q_sc_stat_origin(self, q_sc_key):
        """
        q_sc_key: q_id | SPARQL | tm_comp | tm_value | allow_forever | ord_comp | ord_rank | aggregate
        1. tm_comp: < / == / > / None
        2. tm_value: xxxx_yyyy / (blank)
        3. allow_forever: Fyes / Fdyn / Fhalf / Fno / (blank)
        4. ord_comp: min / max / None
        5. ord_rank: xxx / (blank)
        6. aggregate: Agg / None
        """

        hit = q_sc_key in self.q_sc_dict
        show_request = (not hit or self.vb >= 2)  # going to perform a real query, or just want to show more
        if show_request:
            LogInfo.begin_track('[%s] Q_Schema Request:', show_time())
            LogInfo.logs(q_sc_key)
        if hit and self.vb >= 1:
            if not show_request:
                LogInfo.logs('[%s] Q_Schema hit!', show_time())
            else:
                LogInfo.logs('Q_Schema hit!')
        if hit:
            stat = self.q_sc_dict[q_sc_key]
            spt = stat.split('_')
            ans_size = int(spt[0])
            p, r, f1 = [float(x) for x in spt[1:]]
        else:
            """
            We don't save the detail result of a non-fuzzy SPARQL query in the F1-query scenario,
            that is because we don't expect that a detail SPARQL query occurs many times in different questions.
            If would cost too much memory if we force to save all non-fuzzy query results,
            the main motivation of using cache is to support the client program quickly running multiple times,
            not for saving time ACROSS DIFFERENT QUESTIONS.
            """
            (q_id, sparql_query,
             tm_comp, tm_value, allow_forever,
             ord_comp, ord_rank, agg) = q_sc_key.split('|')
            assert sparql_query.startswith('SELECT DISTINCT ')
            assert ' WHERE ' in sparql_query
            symbol_str = sparql_query[16: sparql_query.find(' WHERE ')]  #找到sql语句中三元组的位置
            ret_symbol_list = symbol_str.split(' ')
            gold_list = self.q_gold_dict[q_id] # 加载每个问题对应的答案字符串
            forbidden_mid_set = self.extract_forbidden_mid(sparql_query=sparql_query) # 加载sql语句中所有的实体
            if self.vb >= 1:
                LogInfo.logs('Forbidden mid: %s', forbidden_mid_set)
            key = self.shrink_query(sparql_query=sparql_query) # 清除掉sparql中实体/关系/属性的多余的语句前缀。
            query_ret = self.kernel_query(key)
            if query_ret is None:   # encountered error, and won't save q_sc_stat.
                ans_size = 0
                p = r = f1 = 0.
            else:
                # predict_value = self.answer_post_process(
                #     forbidden_mid_set=forbidden_mid_set, query_ret=query_ret)
                # 通过sparsql找到最后的答案
                predict_value = self.answer_post_process_new(
                    query_ret=query_ret,
                    forbidden_mid_set=forbidden_mid_set,
                    ret_symbol_list=ret_symbol_list,
                    tm_comp=tm_comp, tm_value=tm_value,
                    allow_forever=allow_forever,
                    ord_comp=ord_comp, ord_rank=ord_rank
                )
                predict_list = sorted(list(predict_value))      # change from set to list
                if agg == 'Agg':                        # manually perform COUNT(*)
                    distint_answers = len(predict_value)
                    predict_list = [str(distint_answers)]   # only one answer: the number of distinct targets.

                ans_size = len(predict_list)
                if q_id.startswith('Webq'):
                    r, p, f1 = compute_f1(gold_list, predict_list)
                else:  # CompQ
                    """
                    1. force lowercase, both gold and predict
                    2. hyphen normalize: -, \u2013, \u2212
                    Won't the affect the values to be stored in the file.
                    """
                    eval_gold_list = [self.compq_answer_normalize(x) for x in gold_list]
                    eval_predict_list = [self.compq_answer_normalize(x) for x in predict_list]
                    r, p, f1 = compute_f1(eval_gold_list, eval_predict_list)
                stat = '%d_%.6f_%.6f_%.6f' % (ans_size, p, r, f1)

                # if self.lock is not None:
                #     self.lock.acquire()
                self.q_sc_dict[q_sc_key] = stat
                self.q_sc_buffer.append((q_sc_key, stat))
                # if self.lock is not None:
                #     self.lock.release()

        ret_info = [ans_size, p, r, f1]
        if show_request:
            LogInfo.logs('Answers = %d, P = %.6f, R = %.6f, F1 = %.6f', ans_size, p, r, f1)
            LogInfo.end_track()
        return ret_info


    def query_q_sc_stat(self, q_sc_key):
        """
        q_sc_key: q_id | SPARQL | tm_comp | tm_value | allow_forever | ord_comp | ord_rank | aggregate
        1. tm_comp: < / == / > / None
        2. tm_value: xxxx_yyyy / (blank)
        3. allow_forever: Fyes / Fdyn / Fhalf / Fno / (blank)
        4. ord_comp: min / max / None
        5. ord_rank: xxx / (blank)
        6. aggregate: Agg / None
        """
        hit = q_sc_key in self.q_sc_dict
        show_request = (not hit or self.vb >= 2)  # going to perform a real query, or just want to show more
        if show_request:
            LogInfo.begin_track('[%s] Q_Schema Request:', show_time())
            LogInfo.logs(q_sc_key)
        if hit and self.vb >= 1:
            if not show_request:
                LogInfo.logs('[%s] Q_Schema hit!', show_time())
            else:
                LogInfo.logs('Q_Schema hit!')
        # import pdb; pdb.set_trace()
        if hit:
            stat = self.q_sc_dict[q_sc_key]
            spt = stat.split('___')
            ans_size = int(spt[0])
            p, r, f1 = [float(x) for x in spt[1:-1]]
            ans_str = spt[-1]
        else:
            """
            We don't save the detail result of a non-fuzzy SPARQL query in the F1-query scenario,
            that is because we don't expect that a detail SPARQL query occurs many times in different questions.
            If would cost too much memory if we force to save all non-fuzzy query results,
            the main motivation of using cache is to support the client program quickly running multiple times,
            not for saving time ACROSS DIFFERENT QUESTIONS.
            """
            (q_id, sparql_query,
             tm_comp, tm_value, allow_forever,
             ord_comp, ord_rank, agg) = q_sc_key.split('|')
            assert sparql_query.startswith('SELECT DISTINCT ')
            assert ' WHERE ' in sparql_query
            symbol_str = sparql_query[16: sparql_query.find(' WHERE ')]
            ret_symbol_list = symbol_str.split(' ')
            gold_list = self.q_gold_dict[q_id]  # self.q_gold_dict存储了所有问题的答案（id,答案）
            forbidden_mid_set = self.extract_forbidden_mid(sparql_query=sparql_query) # 加载sql语句中所有的实体
            if self.vb >= 1:
                LogInfo.logs('Forbidden mid: %s', forbidden_mid_set)
            key = self.shrink_query(sparql_query=sparql_query)
            query_ret = self.kernel_query(key)
            if query_ret is None:   # encountered error, and won't save q_sc_stat.
                ans_size = 0
                p = r = f1 = 0.
            else:
                # predict_value = self.answer_post_process(
                #     forbidden_mid_set=forbidden_mid_set, query_ret=query_ret)
                # 对从知识库中得到的数据进行约束过滤，比如时间“2000”，与知识库中‘2000-02-09’之间的匹配
                predict_value = self.answer_post_process_new(
                    query_ret=query_ret,
                    forbidden_mid_set=forbidden_mid_set,
                    ret_symbol_list=ret_symbol_list,# 返回的schema位置：['?o1', '?n1', '?tm1']
                    tm_comp=tm_comp, tm_value=tm_value,
                    allow_forever=allow_forever,
                    ord_comp=ord_comp, ord_rank=ord_rank
                )
                predict_list = sorted(list(predict_value))      # change from set to list
                if agg == 'Agg':                        # manually perform COUNT(*)
                    distint_answers = len(predict_value)
                    predict_list = [str(distint_answers)]   # only one answer: the number of distinct targets.

                ans_size = len(predict_list)
                if q_id.startswith('Webq'):
                    r, p, f1 = compute_f1(gold_list, predict_list)
                else:  # CompQ
                    """
                    1. force lowercase, both gold and predict
                    2. hyphen normalize: -, \u2013, \u2212
                    Won't the affect the values to be stored in the file.
                    """
                    eval_gold_list = [self.compq_answer_normalize(x) for x in gold_list]
                    eval_predict_list = [self.compq_answer_normalize(x) for x in predict_list]
                    r, p, f1 = compute_f1(eval_gold_list, eval_predict_list)
                    # if(r == 1.0):
                    #     import pdb; pdb.set_trace()
                # if(f1 > 0.3):
                #     import pdb; pdb.set_trace()
                ans_str = '\t'.join(predict_list)
                stat = '%d___%.6f___%.6f___%.6f___%s' % (ans_size, p, r, f1, ans_str)

                # if self.lock is not None:
                #     self.lock.acquire()
                self.q_sc_dict[q_sc_key] = stat
                self.q_sc_buffer.append((q_sc_key, stat))
                # if self.lock is not None:
                #     self.lock.release()

        ret_info = [ans_size, p, r, f1, ans_str]
        if show_request:
            LogInfo.logs('Answers = %d, P = %.6f, R = %.6f, F1 = %.6f', ans_size, p, r, f1)
            LogInfo.end_track()
        return ret_info

    def save_buffer(self):
        # if self.lock is not None:
        #     self.lock.acquire()
        with codecs.open(self.sparql_cache_fp, 'a', 'utf-8') as bw:
            for tup in self.sparql_buffer:
                bw.write(json.dumps(tup))
                bw.write('\n')
        with codecs.open(self.qsc_cache_fp, 'a', 'utf-8') as bw:
            for tup in self.q_sc_buffer:
                bw.write(json.dumps(tup))
                bw.write('\n')
        self.sparql_buffer = []
        self.q_sc_buffer = []
        LogInfo.logs('[%s] buffer saved.', show_time())
        # if self.lock is not None:
        #     self.lock.release()
        return 0


# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)


def main():
    # Create server
    srv_port = int(sys.argv[1])
    cache_dir = sys.argv[2]
    vb = int(sys.argv[3])
    LogInfo.logs('srv_port = %d, vb = %d', srv_port, vb)
    server = SimpleXMLRPCServer(("0.0.0.0", srv_port), requestHandler=RequestHandler)
    server.register_introspection_functions()

    service_inst = QueryService(sparql_cache_fp=cache_dir+'/sparql.cache',
                                qsc_cache_fp=cache_dir+'/q_sc_stat.cache',
                                vb=vb)
    server.register_function(service_inst.query_sparql)
    server.register_function(service_inst.query_q_sc_stat)
    server.register_function(service_inst.save_buffer)
    LogInfo.logs('Functions registered: %s', server.system_listMethods())

    # Run the server's main loop
    LogInfo.logs('Begin serving ... ')
    server.serve_forever()


def test():
    service_inst = QueryService(
        sparql_cache_fp='runnings/acl18_cache/tmp/sparql.cache',
        qsc_cache_fp='runnings/acl18_cache/tmp/q_sc_stat.cache',
        vb=3
    )

    # q_id = 'CompQ_1352'
    # sparql = 'SELECT DISTINCT ?o2 ?n2 ?tm1 ?tm2 ?ord WHERE { ' \
    #          'fb:m.07sz1 fb:law.court.judges ?o1 . ' \
    #          '?o1 fb:law.judicial_tenure.judge ?o2 . ' \
    #          '?o2 fb:type.object.name ?n2 . ' \
    #          '?o1 fb:law.judicial_tenure.from_date ?ord . ' \
    #          'OPTIONAL { ?o1 fb:law.judicial_tenure.from_date ?tm1 . } . ' \
    #          'OPTIONAL { ?o1 fb:law.judicial_tenure.to_date ?tm2 . } . ' \
    #          '}'
    # tm_comp = '=='
    # tm_value = '2009_2009'
    # allow_forever = 'Fno'
    # ord_comp = 'max'
    # ord_rank = '1'
    # agg = 'None'

    # q_id = 'CompQ_1783'
    # sparql = 'SELECT DISTINCT ?o2 ?n2 ?ord WHERE { ' \
    #          'fb:m.01d5z fb:baseball.baseball_team.team_stats ?o1 . ' \
    #          '?o1 fb:baseball.baseball_team_stats.season ?o2 . ' \
    #          '?o1 fb:baseball.baseball_team_stats.wins ?ord . ' \
    #          '?o2 fb:type.object.name ?n2 . } '
    # tm_comp = 'None'
    # tm_value = ''
    # allow_forever = ''
    # ord_comp = 'max'
    # ord_rank = '1'
    # agg = 'None'

    q_id = 'CompQ_1705'
    sparql = 'SELECT DISTINCT ?o1 ?n1 ?ord WHERE { ' \
             'fb:m.06x5s fb:time.recurring_event.instances ?o1 . ' \
             '?o1 fb:sports.sports_championship_event.champion fb:m.05tfm . ' \
             '?o1 fb:type.object.name ?n1 . ' \
             '?o1 fb:time.event.end_date ?ord . ' \
             '}'
    tm_comp = 'None'
    tm_value = ''
    allow_forever = ''
    ord_comp = 'max'
    ord_rank = '1'
    agg = 'None'

    q_sc_key = '|'.join([q_id, sparql,
                         tm_comp, tm_value, allow_forever,
                         ord_comp, ord_rank, agg])
    service_inst.query_q_sc_stat(q_sc_key=q_sc_key)
    

def search_relation_type(service_inst, relation):
    # 功能：搜索每个关系对应的头实体和尾实体的主要类型
    sparql_str = service_inst.query_prefix + ' select * where {?s fb:%s ?o2 .} limit 100' % relation
    query_ret = service_inst.backend.query(sparql_str)
    subject_type = {}
    object_type = {}
    if(query_ret != None):
        for entitys in query_ret:
            if(len(entitys) == 2):
                #*****************计算头实体的类型*********************************************
                sparql_str = service_inst.query_prefix + ' select ?o2 where {fb:%s fb:type.object.type ?o2 .}' % entitys[0]
                query_ret = service_inst.backend.query(sparql_str)
                if(query_ret != None):
                    for entity in query_ret:
                        if(entity[0] not in subject_type):
                            subject_type[entity[0]] = 1
                        else:
                            subject_type[entity[0]] += 1
                #*******************计算尾实体的类型*****************************************
                sparql_str = service_inst.query_prefix + ' select ?o2 where {fb:%s fb:type.object.type ?o2 .}' % entitys[1]
                query_ret = service_inst.backend.query(sparql_str)
                if(query_ret != None):
                    for entity in query_ret:
                        if(entity[0] not in object_type):
                            object_type[entity[0]] = 1
                        else:
                            object_type[entity[0]] += 1
    # else:
    #     import pdb; pdb.set_trace()
    subject_type_list = sorted(subject_type.items(), key = lambda x:x[1], reverse=True)
    object_type_list = sorted(object_type.items(), key=lambda  x:x[1], reverse=True)
    return subject_type_list[0:5], object_type_list[0:5]

def search_relation_subobj_type(service_inst, relation):
    # 功能：搜索每个关系对应的头实体和尾实体的主要类型
    sparql_str = service_inst.query_prefix + ' select * where {?s fb:%s ?o2 .} limit 1000' % relation
    query_ret = service_inst.backend.query(sparql_str)
    subobj_types = {}
    if(query_ret != None):
        for entitys in query_ret:
            if(len(entitys) == 2):
                #*****************计算头实体的类型*********************************************
                sparql_str = service_inst.query_prefix + ' select ?o2 where {fb:%s fb:type.object.type ?o2 .}' % entitys[0]
                sub_types = service_inst.backend.query(sparql_str)
                # print(query_ret)
                #*******************计算尾实体的类型*****************************************
                try:
                    sparql_str = service_inst.query_prefix + ' select ?o2 where {fb:%s fb:type.object.type ?o2 .}' % entitys[1]
                    obj_types = service_inst.backend.query(sparql_str)
                except:
                    import pdb; pdb.set_trace()
                    obj_types = []
                # print(query_ret)
                sub_temp = []
                subobj_temp = []
                if(sub_types != None):
                    for item in sub_types:
                        if(item[0] != 'common.topic'):
                            sub_temp.append(item[0])
                else:
                    sub_temp.append(' ')
                if(obj_types != None):
                    for item in obj_types:
                        if(item[0] != 'common.topic'):
                            for i in range(len(sub_temp)):
                                # import pdb; pdb.set_trace()
                                subobj_temp.append(sub_temp[i] + '\t' + item[0])
                else:
                    for i in range(len(sub_temp)):
                        subobj_temp.append(sub_temp[i] + '\t ')
                for item in subobj_temp:
                    if(item not in subobj_types):
                        subobj_types[item] = 1
                    else:
                        subobj_types[item] += 1
            
    # else:
    #     import pdb; pdb.set_trace()
    subobj_types_list = sorted(subobj_types.items(), key = lambda x: x[1], reverse=True)
    # import pdb; pdb.set_trace()
    return subobj_types_list[0:5]
    

def generate_types():
    import pickle
    service_inst = QueryService(
        sparql_cache_fp='runnings/acl18_cache/tmp/sparql.cache',
        qsc_cache_fp='runnings/acl18_cache/tmp/q_sc_stat.cache',
        vb=3
    )
    f = open('./relation_mid.txt', 'r', encoding = 'utf-8')
    rel2subtype = {}
    rel2objtype = {}
    lines = f.readlines()
    for line in lines:
        print(line)
        line_cut = line.strip().split('\t')
        subject_type_list, object_type_list = search_relation_type(service_inst, line_cut[0])
        temp = []
        for item in subject_type_list:
            temp.append(item[0])
        rel2subtype[line_cut[0]] = temp
        temp = []
        for item in object_type_list:
            temp.append(item[0])
        rel2objtype[line_cut[0]] = temp
    f_subtype = open('./rel2subtype_from1000.pkl', 'wb')
    pickle.dump(rel2subtype, f_subtype)    
    f_objtype = open('./rel2objtype_from1000.pkl', 'wb')
    pickle.dump(rel2objtype, f_objtype) 


def generate_subobj_types():
    import pickle
    service_inst = QueryService(
        sparql_cache_fp='runnings/acl18_cache/tmp/sparql.cache',
        qsc_cache_fp='runnings/acl18_cache/tmp/q_sc_stat.cache',
        vb=3
    )
    f = open('/data2/yhjia/dataset/freebase_relation_only_19341.txt', 'r', encoding = 'utf-8')
    rel2subobjtype = {}
    lines = f.readlines()
    for line in lines:
        print(line)
        line = line.strip()
        # if(line == 'user.zsi_editorial.editorial.comment.comments'):
        subobj_types = search_relation_subobj_type(service_inst, line)
        temp = []
        for item in subobj_types:
            temp.append(item[0])
        rel2subobjtype[line] = temp
    f_subobjtype = open('./rel2subobjtype_combine_from1000.pkl', 'wb')
    pickle.dump(rel2subobjtype, f_subobjtype) 

# 获取关系ID对应的名字和描述
def get_name_description():
    import csv
    service_inst = QueryService(
        sparql_cache_fp='runnings/acl18_cache/tmp/sparql.cache',
        qsc_cache_fp='runnings/acl18_cache/tmp/q_sc_stat.cache',
        vb=3
    )
    f = open('/data2/yhjia/dataset/freebase_relation_only_19341.txt', 'r', encoding = 'utf-8')
    rel2info = []
    lines = f.readlines()
    num_description = 0.0
    # num_name = 0.0
    for line in lines:
        rel = line.strip()
        sparql_str = service_inst.query_prefix + ' select * where {fb:%s fb:type.object.name ?o .}' % rel
        query_ret = service_inst.backend.query(sparql_str)
        temp = [rel, '', '']
        if(query_ret != None):# query_ret格式: [['Number']]
            temp[1] = query_ret[0][0]
        # 搜索对于关系的描述信息
        sparql_str = service_inst.query_prefix + ' select * where {fb:%s fb:common.topic.description ?o .}' % rel
        query_ret = service_inst.backend.query(sparql_str)
        if(query_ret != None):
            try:
                temp[2] = query_ret[0][0]
                num_description += 1
            except:
                print(query_ret)
        rel2info.append(temp)
    f = open('rel2name_description.csv', 'w', encoding = 'utf-8')
    writer_csv = csv.writer(f)
    for item in rel2info:
        writer_csv.writerow(item)
    f.close()
    print('有描述的关系词比例：', num_description, len(lines), num_description / len(lines))

# 获取关系ID对应的别名
def get_alias():
    import csv
    service_inst = QueryService(
        sparql_cache_fp='runnings/acl18_cache/tmp/sparql.cache',
        qsc_cache_fp='runnings/acl18_cache/tmp/q_sc_stat.cache',
        vb=3
    )
    f = open('/data2/yhjia/dataset/freebase_relation_only_19341.txt', 'r', encoding = 'utf-8')
    rel2info = []
    lines = f.readlines()
    
    for line in lines:
        rel = line.strip()
        sparql_str = service_inst.query_prefix + ' select ?a where { ?s fb:base.natlang.property_alias.prop fb:%s . ?s fb:base.natlang.property_alias.alias ?a .}' % rel
        query_ret = service_inst.backend.query(sparql_str)
        temp = [rel, '']
        if(query_ret != None and len(query_ret) > 0):# query_ret格式: [['Number']]
            # import pdb; pdb.set_trace()
            for item in query_ret[0]:
                temp[1] += item + '\t'
        rel2info.append(temp)
    f = open('rel2alias.csv', 'w', encoding = 'utf-8')
    writer_csv = csv.writer(f)
    for item in rel2info:
        writer_csv.writerow(item)
    f.close()

# 给定一个实体，判断其类型
def get_types_of_entity(service_inst, entity):
    sparql_str = service_inst.query_prefix + ' select ?o2 where {fb:%s fb:type.object.type ?o2 .}' % entity
    sub_types = service_inst.backend.query(sparql_str)
    sub_temp = {}
    if(sub_types != None):
        for item in sub_types:
            if(item[0] != 'common.topic'):
                if(item[0] not in sub_temp):
                    sub_temp[item[0]] = 1
                else:
                    sub_temp[item[0]] += 1
    sub_list = sorted(sub_temp.items(), key=lambda x: x[1], reverse=True)
    # print(sub_list)
    # print([item[0] for item in sub_list])
    return [item[0] for item in sub_list[0:]]

# 给定一个尾实体，判断其相关头实体的类型
def get_subtype_of_obj(service_inst, entity):
    sparql_str = service_inst.query_prefix + ' select * where {?s ?p fb:%s.} limit 100' % entity
    query_ret = service_inst.backend.query(sparql_str)
    types = {}
    if(query_ret != None):
        for entitys in query_ret:
            if(len(entitys) == 2):
                sub_type = get_types_of_entity(service_inst, entitys[0]) # 得到实体类型
                for item in sub_type:
                    if(item not in types):
                        types[item] = 1
                    else:
                        types[item] += 1
    types_list = sorted(types.items(), key=lambda x: x[1], reverse=True)
    # print('sub:', types_list)
    # return [item[0] for item in types_list[0:] if(item[1] > 1)]
    return types_list

# 给定一个头实体，判断其相关尾实体的类型
def get_objtype_of_sub(service_inst, entity):
    sparql_str = service_inst.query_prefix + ' select * where {fb:%s ?p ?o.} limit 100' % entity
    query_ret = service_inst.backend.query(sparql_str)
    types = {}
    if(query_ret != None):
        for entitys in query_ret:
            if(len(entitys) == 2):
                obj_type = get_types_of_entity(service_inst, entitys[1]) # 得到实体类型
                for item in obj_type:
                    if(item not in types):
                        types[item] = 1
                    else:
                        types[item] += 1
    types_list = sorted(types.items(), key=lambda x: x[1], reverse=True)
    # print('obj:', types_list)
    # return [item[0] for item in types_list[0:] if(item[1] > 1)]
    return types_list
                
# 获取实体类型预训练的语料
def get_pretrain_types_sequence():
    import csv
    service_inst = QueryService(
        sparql_cache_fp='runnings/acl18_cache/tmp/sparql.cache',
        qsc_cache_fp='runnings/acl18_cache/tmp/q_sc_stat.cache',
        vb=3
    )
    f = open('/data2/yhjia/dataset/freebase_relation_only_19341.txt', 'r', encoding = 'utf-8')
    fout = open('/data2/yhjia/dataset/pretrain_types_100_10_max5.txt', 'w', encoding= 'utf-8')
    lines = f.readlines()
    types_data = {}
    for i, line in enumerate(lines):
        print(i, line)
        
        line = line.strip()
        sparql_str = service_inst.query_prefix + ' select * where {?s fb:%s ?o2 .} limit 100' % line
        query_ret = service_inst.backend.query(sparql_str)
        if(query_ret != None):
            for entitys in query_ret:
                if(len(entitys) == 2):
                    #*****************计算头实体的类型*********************************************
                    sub_type = get_types_of_entity(service_inst, entitys[0])
                    #*******************计算尾实体的类型*****************************************
                    obj_type = get_types_of_entity(service_inst, entitys[1])
                    sub_sub_type = get_subtype_of_obj(service_inst, entitys[0])
                    obj_obj_type = get_objtype_of_sub(service_inst, entitys[1])
                    # import pdb; pdb.set_trace()
                    for sub_sub in sub_sub_type:
                        if(len(sub_type) > 0):
                            for sub in sub_type:
                                if(len(obj_type) > 0):
                                    for obj in obj_type:
                                        if(len(obj_obj_type) > 0):
                                            for obj_obj in obj_obj_type:
                                                types_str = sub_sub + ' ' + sub + ' ' + obj + ' ' + obj_obj
                                                # if(types_str == 'sports.sports_team american_football.football_historical_roster_position'):
                                                #     import pdb; pdb.set_trace()
                                                if(types_str not in types_data):
                                                    types_data[types_str] = 1
                                        else:
                                            types_str = sub_sub + ' ' + sub + ' ' + obj
                                            # if(types_str == 'sports.sports_team american_football.football_historical_roster_position'):
                                            #     import pdb; pdb.set_trace()
                                            if(types_str not in types_data):
                                                types_data[types_str] = 1
                                else:
                                    types_str = sub_sub + ' ' + sub
                                    # if(types_str == 'sports.sports_team american_football.football_historical_roster_position'):
                                    #     import pdb; pdb.set_trace()
                                    if(types_str not in types_data):
                                        types_data[types_str] = 1
        if(len(types_data) > 2000000):
            for item in types_data:
                fout.write(item + '\n')
            types_data = {}
            fout.flush()
    for item in types_data:
        fout.write(item + '\n')
    types_data = {}
    fout.flush()


# 获取实体类型预训练的语料，按照图中结点相互连接形式，一跳相邻结点
def get_pretrain_types_graph():
    import csv
    service_inst = QueryService(
        sparql_cache_fp='runnings/acl18_cache/tmp/sparql.cache',
        qsc_cache_fp='runnings/acl18_cache/tmp/q_sc_stat.cache',
        vb=3
    )
    f = open('/data2/yhjia/dataset/freebase_relation_only_19341.txt', 'r', encoding = 'utf-8')
    fout = open('/data2/yhjia/dataset/pretrain_types_100_10_graph_window6.txt', 'w', encoding= 'utf-8')
    lines = f.readlines()
    types_data = {}
    for k, line in enumerate(lines):
        print(k, line)
        line = line.strip()
        sparql_str = service_inst.query_prefix + ' select * where {?s fb:%s ?o2 .} limit 100' % line
        query_ret = service_inst.backend.query(sparql_str)
        if(query_ret != None):
            for entitys in query_ret:
                if(len(entitys) == 2):
                    #*****************计算头实体的类型相关依赖关系*********************************************
                    entity_type = get_types_of_entity(service_inst, entitys[0])
                    sub_type = get_subtype_of_obj(service_inst, entitys[0])
                    obj_type = get_objtype_of_sub(service_inst, entitys[0])
                    sub_type.extend(obj_type)
                    for one_type in entity_type:
                        i = 0
                        while i < len(sub_type):
                            type_str = one_type + ' ' + ' '.join(sub_type[i:i+6])
                            if(type_str not in types_data):
                                types_data[type_str] = 1
                            # import pdb; pdb.set_trace()
                            i += 6
                    #*******************计算尾实体的类型相关依赖关系*****************************************
                    entity_type = get_types_of_entity(service_inst, entitys[1])
                    sub_type = get_subtype_of_obj(service_inst, entitys[1])
                    obj_type = get_objtype_of_sub(service_inst, entitys[1])
                    sub_type.extend(obj_type)
                    for one_type in entity_type:
                        i = 0
                        while i < len(sub_type):
                            type_str = one_type + ' ' + ' '.join(sub_type[i:i+6])
                            if(type_str not in types_data):
                                types_data[type_str] = 1
                            # import pdb; pdb.set_trace()
                            i += 6
        # if(k == 100):
        #     import pdb; pdb.set_trace()
        if(len(types_data) > 1000000):
            for item in types_data:
                fout.write(item + '\n')
            types_data = {}
            fout.flush()  
    for item in types_data:
        fout.write(item + '\n')
    types_data = {}
    fout.flush()           


# 获取实体类型预训练的数据，该数据不能直接使用，只包含每个类型及其对应的邻接类型
def get_types_info_graph():
    try:
        import csv
        import json
        k_log = 0
        num_file = 0
        service_inst = QueryService(
            sparql_cache_fp='runnings/acl18_cache/tmp/sparql.cache',
            qsc_cache_fp='runnings/acl18_cache/tmp/q_sc_stat.cache',
            vb=3
        )
        f = open('/data2/yhjia/dataset/freebase_relation_only_19341.txt', 'r', encoding = 'utf-8')
        # fout = open('/data2/yhjia/dataset/pretrain_types_200_100_graph.txt', 'w', encoding= 'utf-8')
        lines = f.readlines()
        types_data = []
        entitys_dic = {}
        for k, line in enumerate(lines):
            print(k, line)
            line = line.strip()
            sparql_str = service_inst.query_prefix + ' select * where {?s fb:%s ?o2 .} limit 200' % line
            query_ret = service_inst.backend.query(sparql_str)
            if(query_ret != None):
                for entitys in query_ret:
                    if(len(entitys) == 2):
                        #*****************计算头实体的类型相关依赖关系*********************************************
                        if(entitys[0] not in entitys_dic):
                            entitys_dic[entitys[0]] = 1
                            entity_type = get_types_of_entity(service_inst, entitys[0])
                            sub_obj_types = []
                            sub_obj_types.append(entity_type)
                            sub_obj_types.append(get_subtype_of_obj(service_inst, entitys[0]))
                            sub_obj_types.append(get_objtype_of_sub(service_inst, entitys[0]))
                            # import pdb; pdb.set_trace()
                            types_data.append(sub_obj_types)
                        #*******************计算尾实体的类型相关依赖关系*****************************************
                        if(entitys[1] not in entitys_dic):
                            entitys_dic[entitys[1]] = 1
                            entity_type = get_types_of_entity(service_inst, entitys[1])
                            sub_obj_types = []
                            sub_obj_types.append(entity_type)
                            sub_obj_types.append(get_subtype_of_obj(service_inst, entitys[1]))
                            sub_obj_types.append(get_objtype_of_sub(service_inst, entitys[1]))
                            types_data.append(sub_obj_types)
            # if(k == 100):
            #     import pdb; pdb.set_trace()
            if(len(types_data) > 1000000):
                with open('/data2/yhjia/dataset/types_data_' + str(num_file) + '.json', 'w', encoding='utf-8') as f:
                    num_file += 1
                    json.dump(types_data, f) 
                    types_data = []
                k_log = k
        if(len(types_data) > 0):
            with open('/data2/yhjia/dataset/types_data_' + str(num_file) + '.json', 'w', encoding='utf-8') as f:
                num_file += 1
                json.dump(types_data, f) 
                types_data = []
            k_log = k  
    except:
        print(k, line)
        print(k_log)
        import pdb; pdb.set_trace()
                    

# 生成关系对应的实体类型
def get_sub_obj_type():
    import pickle
    service_inst = QueryService(
        sparql_cache_fp='runnings/acl18_cache/tmp/sparql.cache',
        qsc_cache_fp='runnings/acl18_cache/tmp/q_sc_stat.cache',
        vb=3
    )
    f = open('/data2/yhjia/dataset/freebase_relation_only_19341.txt', 'r', encoding = 'utf-8')
    rel2subtype = {}
    rel2objtype = {}
    lines = f.readlines()
    for line in lines:
        print(line)
        line_cut = line.strip().split('\t')
        if(line_cut[0] not in rel2subtype):
            subject_type_list, object_type_list = search_relation_type(service_inst, line_cut[0])
            temp = []
            try:
                for item in subject_type_list:
                    temp.append(item)
            except:
                temp = []
            rel2subtype[line_cut[0]] = temp
            temp = []
            try:
                for item in object_type_list:
                    temp.append(item)
            except:
                temp = []
            rel2objtype[line_cut[0]] = temp
        # import pdb; pdb.set_trace()
    f_subtype = open('./freebase_rel2subtype_from100.pkl', 'wb')
    pickle.dump(rel2subtype, f_subtype)    
    f_objtype = open('./freebase_rel2objtype_from100.pkl', 'wb')
    pickle.dump(rel2objtype, f_objtype)


if __name__ == '__main__':
    # get_types_info_graph()
    # get_pretrain_types_graph()
    # generate_subobj_types()
    # get_alias()
    # get_name_description()
    get_sub_obj_type()
    # main()
    # test()
    # import pickle
    # service_inst = QueryService(
    #     sparql_cache_fp='runnings/acl18_cache/tmp/sparql.cache',
    #     qsc_cache_fp='runnings/acl18_cache/tmp/q_sc_stat.cache',
    #     vb=3
    # )
    # f = open('./compq_relation_mid.txt', 'r', encoding = 'utf-8')
    # # 在已有部分文件的基础上增加
    # f_subtype = open('./compq_rel2subtype_from100.pkl', 'rb')
    # f_objtype = open('./compq_rel2objtype_from100.pkl', 'rb')
    # rel2subtype = pickle.load(f_subtype)
    # rel2objtype = pickle.load(f_objtype)
    # lines = f.readlines()
    # for line in lines:
    #     print(line)
    #     line_cut = line.strip().split('\t')
    #     if(line_cut[0] not in rel2subtype):
    #         subject_type_list, object_type_list = search_relation_type(service_inst, line_cut[0])
    #         temp = []
    #         for item in subject_type_list:
    #             temp.append(item[0])
    #         rel2subtype[line_cut[0]] = temp
    #         temp = []
    #         for item in object_type_list:
    #             temp.append(item[0])
    #         rel2objtype[line_cut[0]] = temp
    # f_subtype = open('./compq_rel2subtype_from100.pkl', 'wb')
    # pickle.dump(rel2subtype, f_subtype)    
    # f_objtype = open('./compq_rel2objtype_from100.pkl', 'wb')
    # pickle.dump(rel2objtype, f_objtype) 
    # import pdb; pdb.set_trace()   


