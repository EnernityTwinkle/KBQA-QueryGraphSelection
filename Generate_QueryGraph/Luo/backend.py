"""
A module to communicate with a SPARQL HTTP endpoint.
The class uses a connection pool to reuse existing connections for new queries.

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>
"""
from urllib3 import HTTPConnectionPool, Retry
# import logging
# import globals
import csv
from io import StringIO
import json
import time
import traceback

# logger = logging.getLogger(__name__)
from LogUtil import LogInfo


FREEBASE_NS_PREFIX = "http://rdf.freebase.com/ns/"
FREEBASE_SPARQL_PREFIX = "fb"
FREEBASE_NAME_RELATION = "type.object.name"
FREEBASE_KEY_PREFIX = "http://rdf.freebase.com/key/"


def remove_freebase_ns(mid):
    """
    Returns a fully qualified MID, with NS
    prefix and brackets.
    :param mid:
    :return:
    """
    if mid.startswith(FREEBASE_NS_PREFIX):
        return mid[len(FREEBASE_NS_PREFIX):]
    return mid


def normalize_freebase_output(text):
    """Remove starting and ending quotes and the namespace prefix.

    :param text:
    :return:
    """
    if len(text) > 1 and text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return remove_freebase_ns(text)


def filter_results_language(results, language):
    """Remove results that contain a literal with another language.

    Empty language is allowed!
    :param results:
    :param language:
    :return:
    """
    filtered_results = []
    for r in results:
        contains_literal = False
        for k in r.keys():
            if r[k]['type'] == 'literal':
                contains_literal = True
                if 'xml:lang' not in r[k] or \
                    r[k]['xml:lang'] == language or \
                        r[k]['xml:lang'] == '':
                    filtered_results.append(r)
        if not contains_literal:
            filtered_results.append(r)
    return filtered_results


class SPARQLHTTPBackend(object):
    def __init__(self, backend_host,
                 backend_port,
                 backend_url,
                 connection_pool_maxsize=100,
                 cache_enabled=True,        # TODO: edited by Kangqi
                 cache_maxsize=10000,
                 retry=None):
        self.backend_host = backend_host
        self.backend_port = backend_port
        self.backend_url = backend_url
        self.connection_pool = None
        self._init_connection_pool(connection_pool_maxsize,
                                   retry=retry)

        # Caching structures.
        self.cache_enabled = cache_enabled
        self.cache_maxsize = cache_maxsize
        self.cache = {}
        self.cached_elements_fifo = []
        self.num_queries_executed = 0
        self.total_query_time = 0.0

    def _init_connection_pool(self, pool_maxsize, retry=None):
        if not retry:
            # By default, retry on 404 and 503 messages because
            # these seem to happen sometimes, but very rarely.
            retry = Retry(total=5, status_forcelist=[404, 503],
                          backoff_factor=0.2)
        self.connection_pool = HTTPConnectionPool(self.backend_host,
                                                  port=self.backend_port,
                                                  maxsize=pool_maxsize,
                                                  retries=retry)

    @staticmethod
    def init_from_config(config_options):
        """Return an instance with options parsed by a config parser.
        :param config_options:
        :return:
        """

        backend_host = config_options.get('SPARQLBackend', 'backend-host')
        backend_port = config_options.get('SPARQLBackend', 'backend-port')
        backend_url = config_options.get('SPARQLBackend', 'backend-url')
        LogInfo.logs('Using SPARQL backend at %s:%s%s', backend_host, backend_port, backend_url)
        return SPARQLHTTPBackend(backend_host, backend_port, backend_url)

    def query_json(self, query, method='GET',
                   normalize_output=normalize_freebase_output,
                   filter_lang='en', vb=0):
        """ Returns the result table of the query as a list of rows. """
        """ Seems deprecated """
        params = {
            # "default-graph-URI": "<http://freebase.com>",
            "query": query,
            "maxrows": 2097151,
            # "debug": "off",
            "timeout": 100,
            "format": "application/json",
            # "save": "display",
            # "fname": ""
        }
        if self.cache_enabled and query in self.cache:
            if vb > 0:
                LogInfo.logs('Return result from cache.')
            return self.cache[query]
        start = time.time()
        resp = self.connection_pool.request(method,
                                            self.backend_url,
                                            fields=params)
        self.total_query_time += (time.time() - start)
        self.num_queries_executed += 1

        try:
            if resp.status == 200:
                data = json.loads(resp.data, encoding='utf-8')
                results = data['results']['bindings']
                if filter_lang:
                    results = filter_results_language(results, filter_lang)
                result_rows = []
                keys = sorted(data['head']['vars'])
                for row in results:
                    result_row = []
                    for k in keys:
                        if k in row:
                            result_row.append(row[k]['value'])
                    result_rows.append(result_row)
                results = [[normalize_output(c) for c in r]
                           for r in result_rows]
            else:
                if vb > 0:
                    LogInfo.logs('Return code %s for query [%s]', resp.status, query)
                    LogInfo.logs('Message: %s', resp.data)
                results = None
        except ValueError:
            if vb > 0:
                LogInfo.begin_track('Error executing query [%s]', query)
                LogInfo.logs(traceback.format_exc())
                LogInfo.logs('Headers: %s', resp.headers)
                LogInfo.logs('Data: %s', resp.data)
                LogInfo.end_track()
            results = None
        # Add result to cache.
        if self.cache_enabled:
            self._add_result_to_cache(query, results)
        return results

    def query(self, query, method='GET',
              normalize_output=normalize_freebase_output,
              parse_safe=False, vb=0):
        """
        Execute SPARQL query.
        Returns a list of elements. Each list element is a list of requested
        columns, in the order requested.
        Literals do not contain a language suffix!
        If an error occurred, returns None.
        """
        # LogInfo.logs('Inside backend: [%s] [vb=%d]', query, vb)
        if self.cache_enabled and query in self.cache:
            if vb > 0:
                LogInfo.logs('Return result from cache.')
            return self.cache[query]
        result_format = "text/tab-separated-values"
        if parse_safe:
            result_format = "text/csv"
        # Construct parameter dict.
        params = {
            # "default-graph-URI": "<http://freebase.com>",
            "query": query,
#             "query": 'PREFIX fb: <http://rdf.freebase.com/ns/>\
# SELECT DISTINCT * WHERE { fb:m.07484 fb:people.person.date_of_birth ?o1 . OPTIONAL { ?o1 fb:common.topic.notable_types ?type1 .} }',
            "maxrows": 98931,
            # "debug": "off",
            "timeout": 100,      # 10 seconds
            "format": result_format,
            # "save": "display",
            # "fname": ""
        }
        start = time.time()
        resp = self.connection_pool.request(method, self.backend_url, fields=params)
        self.total_query_time += (time.time() - start)
        self.num_queries_executed += 1
        try:
            if resp.status == 200:
                text = resp.data
                # import pdb; pdb.set_trace()
                # Use csv module to parse
                if parse_safe:
                    data = csv.reader(StringIO.StringIO(text))
                    # Consume header.
                    fields = data.next()
                    results = [[normalize_output(c.decode('utf8')) for c in resp]
                               for resp in data]
                else:
                    results = [[normalize_freebase_output(c) for c in
                                l.split('\t')]
                               for l in text.decode('utf8').split('\n') if l][1:]
            else:
                if vb > 0:
                    LogInfo.logs("Return code %s for query '%s'", resp.status, query)
                    LogInfo.logs("Message: %s", resp.data)
                else:
                    LogInfo.logs('Return code %s.', resp.status)
                results = None
        except ValueError:
            if vb > 0:
                LogInfo.begin_track('Error executing query [%s]', query)
                LogInfo.logs(traceback.format_exc())
                LogInfo.logs('Headers: %s', resp.headers)
                LogInfo.logs('Data: %s', resp.data)
                LogInfo.end_track()
            else:
                LogInfo.logs('ValueError exception found!')
            results = None
        # Add result to cache.
        if self.cache_enabled:
            self._add_result_to_cache(query, results)
        return results

    def query_more(self, query, method='GET',
              normalize_output=normalize_freebase_output,
              parse_safe=False, vb=0):
        """
        Execute SPARQL query.
        Returns a list of elements. Each list element is a list of requested
        columns, in the order requested.
        Literals do not contain a language suffix!
        If an error occurred, returns None.
        与query函数的区别：能够得到更多的信息，具体可参照html返回结果，如datatype
        """
        # LogInfo.logs('Inside backend: [%s] [vb=%d]', query, vb)
        if self.cache_enabled and query in self.cache:
            if vb > 0:
                LogInfo.logs('Return result from cache.')
            return self.cache[query]
        result_format = "json"
        if parse_safe:
            result_format = "text/csv"
        # Construct parameter dict.
        params = {
            # "default-graph-URI": "<http://freebase.com>",
            "query": query,
#             "query": 'PREFIX fb: <http://rdf.freebase.com/ns/>\
# select distinct ?answer ?answer_name where {?09sq8_9 fb:base.yupgrade.user.topics fb:m.03cm9b . ?09sq8_9 fb:base.yupgrade.user.topics ?answer .  OPTIONAL {?answer fb:type.object.name ?answer_name .} }',
            "maxrows": 98931,
            # "debug": "off",
            "timeout": 100,      # 10 seconds
            "format": result_format,
            # "save": "display",
            # "fname": ""
        }
        start = time.time()
        resp = self.connection_pool.request(method, self.backend_url, fields=params)
        self.total_query_time += (time.time() - start)
        self.num_queries_executed += 1
        try:
            # import pdb; pdb.set_trace()
            if resp.status == 200:
                text = resp.data
                json_text = json.loads(text)
                vars_head = json_text['head']['vars']
                # import pdb; pdb.set_trace()
                bindings = json_text['results']['bindings']
                results = []
                for item in bindings:
                    res_temp = []
                    for key in vars_head:
                        if(key in item):
                            if('type' in key):
                                res_temp.append(item[key]['value'].split('/')[-1])
                            else:
                                res_temp.append(item[key]['value'])
                        else:
                            if('type' in key):# 判断是否是实体类型schema
                                # print(bindings)
                                # import pdb; pdb.set_trace()
                                if('datatype' in item['o' + key[-1]]):
                                    res_temp.append(item['o' + key[-1]]['datatype'].split('#')[-1])# 'datatype': 'http://www.w3.org/2001/XMLSchema#datetime'
                                else:
                                    res_temp.append('')
                    results.append(res_temp)
                    # print('results:', results)
            else:
                if vb > 0:
                    LogInfo.logs("Return code %s for query '%s'", resp.status, query)
                    LogInfo.logs("Message: %s", resp.data)
                else:
                    print('query:', query)
                    LogInfo.logs('Return code %s.', resp.status)
                    # import pdb; pdb.set_trace()
                results = None
        except ValueError:
            if vb > 0:
                LogInfo.begin_track('Error executing query [%s]', query)
                LogInfo.logs(traceback.format_exc())
                LogInfo.logs('Headers: %s', resp.headers)
                LogInfo.logs('Data: %s', resp.data)
                LogInfo.end_track()
            else:
                LogInfo.logs('ValueError exception found!')
            results = None
        # Add result to cache.
        if self.cache_enabled:
            self._add_result_to_cache(query, results)
        return results

    def _add_result_to_cache(self, query, result):
        self.cached_elements_fifo.append(query)
        self.cache[query] = result
        if len(self.cached_elements_fifo) > self.cache_maxsize:
            to_delete = self.cached_elements_fifo.pop(0)
            del self.cache[to_delete]

    def paginated_query(self, query, page_size=1040000, **kwargs):
        """A generator for a paginated query.

        :param query:
        :param page_size:
        :param kwargs:
        :return:
        """
        offset = 0
        limit_query = u"%s LIMIT %s OFFSET %s" % (query, page_size, offset)
        result = self.query(limit_query, **kwargs)
        while result:
            yield result
            if len(result) < page_size:
                break
            offset += page_size
            limit_query = u"%s LIMIT %s OFFSET %s" % (query, page_size, offset)
            result = self.query(limit_query, **kwargs)
        raise StopIteration()


#  def main():
    #  sparql = SPARQLHTTPBackend('202.120.38.146', '8999', '/sparql')
    #  query = '''
    #  PREFIX fb: <http://rdf.freebase.com/ns/>
    #  SELECT DISTINCT ?x
    #  WHERE {
     #  ?s fb:type.object.name "Albert Einstein"@EN .
     #  ?s ?p ?o .
     #  FILTER regex(?p, "profession") .
     #  ?o fb:type.object.name ?x .
     #  FILTER (LANG(?x) = "en") }
    #  '''
    #  print(sparql.query(query))
    #  query = '''
        #  SELECT ?name where {
        #  ?x <http://rdf.freebase.com/ns/type.object.name> ?name.
        #  FILTER (lang(?name) != "en")
        #  } LIMIT 100
    #  '''
    #  print(sparql.query(query))


def main():
    sparql = SPARQLHTTPBackend('192.168.126.139', '8999', '/sparql')
    # query = '''
    # PREFIX fb: <http://rdf.freebase.com/ns/>
    # SELECT DISTINCT ?x
    # WHERE {
     # ?s fb:type.object.name "Albert Einstein"@EN .
     # ?s ?p ?o .
     # FILTER regex(?p, "profession") .
     # ?o fb:type.object.name ?x .
     # FILTER (LANG(?x) = "en") }
    # '''
    query = '''
    PREFIX fb: <http://rdf.freebase.com/ns/>
    select distinct ?o where {?s fb:type.object.name ?o} limit 100
    '''
    print(sparql.query(query))


if __name__ == '__main__':
    main()
