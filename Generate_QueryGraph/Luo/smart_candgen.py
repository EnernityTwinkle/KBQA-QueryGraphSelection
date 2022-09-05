"""
Author: Kangqi Luo
Date: 180209
Goal: More efficient candgen based on fixed S-MART data
"""

import os
import json
import codecs
import shutil
import argparse
from global_linker import GlobalLinker, LinkData
from cand_searcher import CandidateSearcher
from query_service import QueryService

from u import load_webq, load_compq
from word_emb import WordEmbeddingUtil

from LogUtil import LogInfo
from time_track import TimeTracker as Tt

# import pdb; pdb.set_trace()
parser = argparse.ArgumentParser(description='Efficient Candidate Generation')
parser.add_argument('--machine', default='Blackhole', choices=['Blackhole', 'Darkstar'])
parser.add_argument('--word_emb', default='glove')
parser.add_argument('--dim_emb', type=int, default=300, help='word/predicate embedding dimension')
parser.add_argument('--data_name', default='WebQ', choices=['WebQ', 'CompQ', 'SimpQ'])
parser.add_argument('--lex_name', default='s-mart', choices=['s-mart', 'xs-lex', 'emsamble_data'])
# parser.add_argument('--lex_name', default='s-mart')
parser.add_argument('--group_idx', type=int)
parser.add_argument('--allow_forever', default='Fhalf')
parser.add_argument('--linking_only', action='store_true')
parser.add_argument('--simple_type', action='store_true')
parser.add_argument('--simple_time', action='store_true')
parser.add_argument('--output_dir')
parser.add_argument('--verbose', type=int, default=1)


def construct_conflict_matrix(gather_linkings):
    gl_size = len(gather_linkings)
    conflict_matrix = []
    for i in range(gl_size):
        local_conf_list = []
        for j in range(gl_size):
            if i == j:
                local_conf_list.append(j)
            elif is_overlap(gather_linkings[i], gather_linkings[j]):
                local_conf_list.append(j)
            elif gather_linkings[i].category == gather_linkings[j].category == 'Type':
                local_conf_list.append(j)
            elif gather_linkings[i].category == gather_linkings[j].category == 'Ordinal':
                local_conf_list.append(j)
        conflict_matrix.append(local_conf_list)
    return conflict_matrix


def is_overlap(gl_a, gl_b):
    if gl_a.end <= gl_b.start:
        return False
    if gl_b.end <= gl_a.start:
        return False
    return True


class SMARTCandidateGenerator:

    def __init__(self, data_name, lex_name, wd_emb_util, query_srv, allow_forever,
                 simple_type_match, simple_time_match, vb=1):
        # query_srv could be the real instance, or a proxy
        self.data_name = data_name
        assert data_name in ('CompQ', 'WebQ')
        q_links_fp = ''
        if data_name == 'WebQ':
            q_links_fp = '../../data/linkings/WebQ/webq.all.top3_source_ensemble_data.q_links'
            # q_links_fp = '/data3/jzzhou/research-code/acl-1209-amr-kbqa/aaai20-qa-junhui/entity-linking/linkings/WebQ/webq.all.%s.q_links' % lex_name
        elif data_name == 'CompQ':
            q_links_fp = '../../data/linkings/CompQ/compQ.all.top5_source_ensemble_data.q_links'
            # q_links_fp = './data/linkings/CompQ/webq.all.%s.q_links' % lex_name
            # q_links_fp = '/home/data/ComplexQuestions/ACL18/compQ.all.%s.q_links' % lex_name

        self.query_srv = query_srv
        self.global_linker = GlobalLinker(wd_emb_util=wd_emb_util, q_links_fp=q_links_fp)   # 加载smart数据，生成type embbeding
        # import pdb; pdb.set_trace()
        self.cand_searcher = CandidateSearcher(query_service=query_srv,
                                               wd_emb_util=wd_emb_util, vb=vb)
        self.allow_forever = allow_forever      # control the strategy of time constraint matching
        self.simple_type_match = simple_type_match  # True: won't perform any type filtering
        self.simple_time_match = simple_time_match  # True: won't consider interval matching
        LogInfo.logs('simple type match = %s', self.simple_type_match)
        LogInfo.logs('simple time match = %s', self.simple_time_match)
        self.vb = vb
        self.vb = 2
        # vb = 0: show basic flow of the process
        # vb = 1: show detail linking information
        # vb = 2: show schemas & F1 query detail
        # import pdb; pdb.set_trace()

    def add_type_for_answer(self):
        '''
        功能：为候选路径对应的答案节点增加类型信息
        输入：
        '''

    def single_question_candgen(self, q_idx, qa, link_fp, opt_sc_fp, linking_only):
        # =================== Linking first ==================== #
        Tt.start('linking')   # focus linking
        if os.path.isfile(link_fp):
            gather_linkings = []
            with codecs.open(link_fp, 'r', 'utf-8') as br:
                for line in br.readlines():
                    tup_list = json.loads(line.strip())
                    ld_dict = {k: v for k, v in tup_list}
                    gather_linkings.append(LinkData(**ld_dict))
            LogInfo.logs('Read %d links from file.', len(gather_linkings))
        else:
            # import pdb; pdb.set_trace()
            tok_list = [tok for tok in qa['tokens']]
            gather_linkings = self.global_linker.perform_linking(q_idx=q_idx, tok_list=tok_list)# 得到各种链接后的信息
        # import pdb; pdb.set_trace()
        """Step 1: Focus linking."""
        el_size = len(list(filter(lambda x: x.category == 'Entity', gather_linkings)))   # from stagg results
        tl_size = len(list(filter(lambda x: x.category == 'Type', gather_linkings)))     # calculate uni/bi/tri-tokens similarity with type embedding
        tml_size = len(list(filter(lambda x: x.category == 'Time', gather_linkings)))    # mark (before, <), (after, >), (since, >=)
        ord_size = len(list(filter(lambda x: x.category == 'Ordinal', gather_linkings))) # consider 3 conditions xx-th: first, 7th, most ..., single -est:longest, xx-th -est combine: second largest
        LogInfo.begin_track('Show %d E + %d T + %d Tm + %d Ord = %d linkings:',
                            el_size, tl_size, tml_size, ord_size, len(gather_linkings))
        if self.vb >= 1:
            for gl in gather_linkings:
                LogInfo.logs(gl.display())
        LogInfo.end_track()
        Tt.record('linking')
        # import pdb; pdb.set_trace()
        # ==================== Save linking results, the same result as kangqi ================ #
        if not os.path.isfile(link_fp):
            with codecs.open(link_fp + '.tmp', 'w', 'utf-8') as bw:
                for gl in gather_linkings:
                    bw.write(json.dumps(gl.serialize()) + '\n')
            shutil.move(link_fp + '.tmp', link_fp)
            LogInfo.logs('%d link data save to file.', len(gather_linkings))
        if linking_only:
            return
        # ===================== Prepare necessary states for linking =============== #
        q_id = '%s_%d' % (self.data_name, q_idx)
        conflict_matrix = construct_conflict_matrix(gather_linkings)
        entity_linkings = list(filter(lambda x: x.category == 'Entity', gather_linkings))
        type_linkings = list(filter(lambda x: x.category == 'Type', gather_linkings))
        time_linkings = list(filter(lambda x: x.category == 'Time', gather_linkings))
        ordinal_linkings = list(filter(lambda x: x.category == 'Ordinal', gather_linkings))
        # ============================== Searching start ========================= #
        aggregate = False
        """ 180308: always ignore aggregation, since we've found the fucked up data distribution """
        # lower_tok_list = [tok.token.lower() for tok in qa['tokens']]
        # lower_tok_str = ' '.join(lower_tok_list)
        # aggregate = lower_tok_str.startswith('how many ')
        # # apply COUNT(*) to all candidate schemas if we found "how many" at the beginning of a question
        """ ================ Step 1: coarse linking, using entities only ================ """
        LogInfo.begin_track('Coarse level searching (total entities = %d):', el_size)
        entity_linkings.sort(key=lambda _el: _el.value)
        """ We sort the linking data, for identifying potential duplicate SPARQL queries and saving time. """
        Tt.start('coarse_comb')
        # import pdb; pdb.set_trace()
        coarse_states = self.cand_searcher.find_coarse_schemas(
            entity_linkings=entity_linkings, conflict_matrix=conflict_matrix, aggregate=aggregate)
        # [(schema, visit_arr)]
        LogInfo.logs('%d coarse schemas retrieved from scratch.', len(coarse_states))
        Tt.record('coarse_comb')
        # import pdb; pdb.set_trace()
        coarse_states = self.batch_schema_f1_query(q_id=q_id, states=coarse_states, level='coarse')
        LogInfo.end_track('Coarse level ended, resulting in %d schemas.', len(coarse_states))
        ### 经过coarse linking，得到每个粗粒度的实体链接的结果，以及每个实体对应的图结构的查询答案，以及每个实体对应图结构的f1\p\r值
        """ ================ Step 2: adding type information ================ """
        LogInfo.begin_track('Type level searching (total types = %d):', tl_size)
        Tt.start('typed_comb')
        typed_states = []
        for idx, (coarse_sc, visit_arr) in enumerate(coarse_states):
            if idx % 100 == 0:
                LogInfo.logs('Current: %d / %d', idx, len(coarse_states))
            typed_states += self.cand_searcher.find_typed_schemas(
                type_linkings=type_linkings, conflict_matrix=conflict_matrix,
                start_schema=coarse_sc, visit_arr=visit_arr,
                simple_type_match=self.simple_type_match
            )
        # import pdb; pdb.set_trace()
        LogInfo.logs('%d typed schemas retrieved from %d coarse schemas.', len(typed_states), len(coarse_states))
        Tt.record('typed_comb')
        typed_states = self.batch_schema_f1_query(q_id=q_id, states=typed_states, level='typed')
        typed_states = coarse_states + typed_states     # don't forget those schemas without type constraints
        LogInfo.end_track('Typed level ended, resulting in %d schemas.', len(typed_states))
        # typed_states = coarse_states
        """ ================ Step 3: adding time information ================ """
        LogInfo.begin_track('Time level searching (total times = %d):', tml_size)
        Tt.start('timed_comb')
        timed_states = []
        for idx, (typed_sc, visit_arr) in enumerate(typed_states):
            if idx % 100 == 0:
                LogInfo.logs('Current: %d / %d', idx, len(typed_states))
            timed_states += self.cand_searcher.find_timed_schemas(
                time_linkings=time_linkings, conflict_matrix=conflict_matrix,
                start_schema=typed_sc, visit_arr=visit_arr,
                simple_time_match=self.simple_time_match    # True: degenerates into Bao
            )
        LogInfo.logs('%d timed schemas retrieved from %d typed schemas.', len(timed_states), len(typed_states))
        Tt.record('timed_comb')
        # import pdb; pdb.set_trace()
        timed_states = self.batch_schema_f1_query(q_id=q_id, states=timed_states, level='timed')
        # import pdb; pdb.set_trace()
        timed_states = typed_states + timed_states      # don't forget the previous schemas
        LogInfo.end_track('Time level ended, resulting in %d schemas.', len(timed_states))
        
        """ ================ Step 4: ordinal information as the final step ================ """
        LogInfo.begin_track('Ordinal level searching (total ordinals = %d):', ord_size)
        Tt.start('ord_comb')
        final_states = []
        for idx, (timed_sc, visit_arr) in enumerate(timed_states):
            if idx % 100 == 0:
                LogInfo.logs('Current: %d / %d', idx, len(timed_states))
            final_states += self.cand_searcher.find_ordinal_schemas(ordinal_linkings=ordinal_linkings,
                                                                    start_schema=timed_sc,
                                                                    visit_arr=visit_arr)
        LogInfo.logs('%d ordinal schemas retrieved from %d timed schemas.', len(final_states), len(timed_states))
        Tt.record('ord_comb')
        final_states = self.batch_schema_f1_query(q_id=q_id, states=final_states, level='ordinal')
        final_states = timed_states + final_states
        LogInfo.end_track('Ordinal level ended, we finally collected %d schemas.', len(final_states))
        # import pdb; pdb.set_trace()
        final_schemas = [tup[0] for tup in final_states]
        self.query_srv.save_buffer()

        '''===================为答案节点增加类型信息================================'''
        for sc in final_schemas:
            ans_str = sc.ans_str
            ans_type_prop = 'common.topic.notable_types' # 表示选取的是common.topic.notable_types对应的类型标签
            sparql_str = sc.build_sparql(simple_time_match=self.simple_time_match, ans_type=ans_type_prop)
            ans_list = ans_str.split('\t')
            type_dic = {}
            for ans in ans_list:
                sparql_str = sparql_str.replace('?n%d' % sc.hops, '"%s"@en' % ans)
                # print(sparql_str)
                ans_type = self.query_srv.backend.query_more(self.query_srv.query_prefix + sparql_str)
                if(ans_type != None):
                    for item in ans_type[0:100]:
                        # print(len(item))
                        if(item[-1] not in type_dic):
                            type_dic[item[-1]] = 1
                        else:
                            type_dic[item[-1]] += 1
            type_list = sorted(type_dic.items(), key=lambda x:x[1], reverse=True)[0:5]
            # print(type_list)
            if(len(type_list) == 0 or type_list[0][0] == ''):
                ans_type_prop = 'type.object.type' # 表示选取的是type.object.type对应的类型标签
                sparql_str = sc.build_sparql(simple_time_match=self.simple_time_match, ans_type=ans_type_prop)
                ans_list = ans_str.split('\t')
                type_dic = {}
                for ans in ans_list:
                    sparql_str = sparql_str.replace('?n%d' % sc.hops, '"%s"@en' % ans)
                    # print(sparql_str)
                    ans_type = self.query_srv.backend.query_more(self.query_srv.query_prefix + sparql_str)
                    if(ans_type != None):
                        for item in ans_type[0:100]:
                            # print(len(item))
                            if(item[-1] not in type_dic):
                                type_dic[item[-1]] = 1
                            else:
                                type_dic[item[-1]] += 1
                type_list = sorted(type_dic.items(), key=lambda x:x[1], reverse=True)[0:5]
            # print(type_list)
            # import pdb; pdb.set_trace()
            sc.answer_type = '\t'.join([item[0] for item in type_list])
            # import pdb; pdb.set_trace()

        # ==================== Save schema results ================ #
        # p, r, f1, ans_size, hops, raw_paths, (agg)
        # raw_paths: (category, gl_pos, gl_mid, pred_seq)
        with codecs.open(opt_sc_fp + '.tmp', 'w', 'utf-8') as bw:
            for sc in final_schemas:
                sc_info_dict = {k: getattr(sc, k) for k in ('p', 'r', 'f1', 'ans_size', 'hops', 'ans_str', 'answer_type')}
                if sc.aggregate is not None:
                    sc_info_dict['agg'] = sc.aggregate
                opt_raw_paths = []
                for cate, gl, pred_seq in sc.raw_paths:
                    opt_raw_paths.append((cate, gl.gl_pos, gl.value, pred_seq))
                sc_info_dict['raw_paths'] = opt_raw_paths
                bw.write(json.dumps(sc_info_dict) + '\n')
        shutil.move(opt_sc_fp + '.tmp', opt_sc_fp)
        LogInfo.logs('%d schemas successfully saved into [%s].', len(final_schemas), opt_sc_fp)
        # import pdb; pdb.set_trace()
        del coarse_states
        del typed_states
        del timed_states
        del final_states

    def batch_schema_f1_query(self, q_id, states, level):
        """
        perform F1 query for each schema in the state.
        :param q_id: WebQ-xxx / CompQ-xxx
        :param states: [(schema, visit_arr)]
        :param level: coarse / typed / timed / ordinal
        :return: filtered states where each schema returns at least one answer.
        """
        Tt.start('%s_F1' % level)
        LogInfo.begin_track('Calculating F1 for %d %s schemas:', len(states), level)
        for idx, (sc, _) in enumerate(states):
            if idx % 100 == 0:
                LogInfo.logs('Current: %d / %d', idx, len(states))
            sparql_str = sc.build_sparql(simple_time_match=self.simple_time_match)
            # import pdb; pdb.set_trace()
            tm_comp, tm_value, ord_comp, ord_rank, agg = sc.build_aux_for_sparql()
            allow_forever = self.allow_forever if tm_comp != 'None' else ''
            # won't specific forever if no time constraints
            q_sc_key = '|'.join([q_id, sparql_str,
                                 tm_comp, tm_value, allow_forever,
                                 ord_comp, ord_rank, agg])
            if self.vb >= 2:
                LogInfo.begin_track('Checking schema %d / %d:', idx, len(states))
                LogInfo.logs(sc.disp_raw_path())
                LogInfo.logs('var_types: %s', sc.var_types)
                LogInfo.logs(sparql_str)
            Tt.start('query_q_sc_stat')
            # import pdb; pdb.set_trace()
            sc.ans_size, sc.p, sc.r, sc.f1, sc.ans_str = self.query_srv.query_q_sc_stat(q_sc_key)
            Tt.record('query_q_sc_stat')
            if self.vb >= 2:
                LogInfo.logs('Answers = %d, P = %.6f, R = %.6f, F1 = %.6f', sc.ans_size, sc.p, sc.r, sc.f1)
                LogInfo.end_track()
            # import pdb; pdb.set_trace()
        filt_states = list(filter(lambda _tup: _tup[0].ans_size > 0, states))
        LogInfo.end_track('%d / %d %s schemas kept with ans_size > 0.', len(filt_states), len(states), level)
        Tt.record('%s_F1' % level)
        return filt_states


def main(args):
    assert args.data_name in ('WebQ', 'CompQ')
    if args.data_name == 'WebQ':
        qa_list = load_webq()
        # qa_list中每个元素格式:{'utterance': 'what is the name of justin bieber brother?', 'targetValue': ['Jazmyn Bieber', 'Jaxon Bieber']}
        # import pdb; pdb.set_trace()
    else:
        qa_list = load_compq()
    print('args.linking_only:', args.linking_only)
    if args.linking_only:
        query_srv = None
        q_start = 0
        q_end = len(qa_list)
        import pdb; pdb.set_trace()
    else:
        group_idx = args.group_idx
        q_start = group_idx * 100
        q_end = group_idx * 100 + 100
        if args.data_name == 'CompQ':
            sparql_cache_fp = 'runnings/compq_cache/group_cache/sparql.g%02d.cache' % group_idx
            q_sc_cache_fp = 'runnings/compq_cache/group_cache/q_sc_stat.g%02d.cache' % group_idx
        else:
            sparql_cache_fp = 'runnings/webq_cache/group_cache_%s/sparql.g%02d.cache' % (args.data_name, group_idx)
            q_sc_cache_fp = 'runnings/webq_cache/group_cache_%s/q_sc_stat.g%02d.cache' % (args.data_name, group_idx)
        query_srv = QueryService(
            sparql_cache_fp=sparql_cache_fp,
            qsc_cache_fp=q_sc_cache_fp, vb=1
        )
    # import pdb; pdb.set_trace()
    # 加载各种词向量
    wd_emb_util = WordEmbeddingUtil(wd_emb=args.word_emb,
                                    dim_emb=args.dim_emb)
    # import pdb; pdb.set_trace()
    # 加载各种实体链接结果，包括序数词、最高级以及类型词等,并用于候选生成
    smart_cand_gen = SMARTCandidateGenerator(data_name=args.data_name,
                                             lex_name=args.lex_name,
                                             wd_emb_util=wd_emb_util,
                                             query_srv=query_srv,
                                             allow_forever=args.allow_forever,
                                             vb=args.verbose,
                                             simple_type_match=args.simple_type,
                                             simple_time_match=args.simple_time)
    # import pdb; pdb.set_trace()
    all_list_fp = []
    for q_idx, qa in enumerate(qa_list):
        if q_idx < q_start or q_idx >= q_end:
            continue
        # if q_idx != 1302:
        #     continue
        LogInfo.begin_track('Entering Q %d / %d [%s]:', q_idx, len(qa_list), \
            qa['utterance'].encode('utf-8'))
        sub_idx = q_idx // 100 * 100
        sub_dir = '%s/data/%04d-%04d' % (args.output_dir, sub_idx, sub_idx + 99)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        # save_ans_fp = '%s/%d_ans' % (sub_dir, q_idx)
        opt_sc_fp = '%s/%04d_schema' % (sub_dir, q_idx)
        link_fp = '%s/%04d_links' % (sub_dir, q_idx)
        all_list_fp.append('/'.join(opt_sc_fp.split('/')[-3:]))
        if os.path.isfile(opt_sc_fp):
            LogInfo.end_track('Skip this question, already saved.')
            continue
        Tt.start('single_q')
        # import pdb; pdb.set_trace()
        smart_cand_gen.single_question_candgen(q_idx=q_idx, qa=qa,
                                               link_fp=link_fp,
                                               opt_sc_fp=opt_sc_fp,
                                               linking_only=args.linking_only)
        Tt.record('single_q')
        LogInfo.end_track()     # End of Q
        # import pdb; pdb.set_trace()
    #  assert len(all_list_fp) == 5810
    with open(f'{args.output_dir}/all_list', 'a') as fp:
        if len(all_list_fp) > 0:
            fp.write('\n'.join(all_list_fp)+'\n')
    # import pdb; pdb.set_trace()


if __name__ == '__main__':
    LogInfo.begin_track('[smart_candgen] ... ')
    _args = parser.parse_args()
    main(_args)
    LogInfo.end_track('All Done.')
    Tt.display()
