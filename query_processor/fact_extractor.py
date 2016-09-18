import logging
import os
import modules
from util import readFile, writeFile

logger = logging.getLogger(__name__)

class FactExtractor(object):

    def __init__(self, fact_list_dir):
        if not os.path.exists(fact_list_dir):
            os.makedirs(fact_list_dir)
        self.fact_list_dir = fact_list_dir
        self.backend = modules.sparql_backend

        self.entities_with_alias_query = '''
        PREFIX fb: <http://rdf.freebase.com/ns/>
        SELECT ?s ?name where {
            ?s fb:common.topic.alias ?o .
            FILTER (lcase(str(?o)) = "%s") .
            ?s fb:type.object.name ?name
        }
        '''

        self.name_by_id_query = '''
        PREFIX fb: <http://rdf.freebase.com/ns/>
        SELECT ?o where {
            fb:%s fb:type.object.name ?o
        }
        '''


    @staticmethod
    def init_from_config(args, config_options):
        fact_list_dir = config_options.get('FactList',
                                        args.dataset)
        return FactExtractor(fact_list_dir)

    def extract_fact_list(self, query):
        if self.fact_list_on_disk(query):
            return self.load_fact_list_from_disk(query)
        else:
            question = query.utterance.lower()
            tokens = question.split(' ')

    def extract_fact_list_with_tokens(self, tokens):
        if len(tokens) == 0:
            return []
        if len(tokens) == 1:
            token = tokens[0]
            entities = self.backend.query(self.entities_with_alias_query % token)

    def extract_fact_list_with_str(self, q):
        entities = self.backend.query(self.entities_with_alias_query % q)
        result = []
        for x in entities:
            s, s_name = x[0], x[1]


    def fact_list_on_disk(self, query):
        id = query.id
        file_path = self.fact_list_dir + str(id)
        return os.path.isfile(file_path)

    def load_fact_list_from_disk(self, query):
        id = query.id
        file_path = self.fact_list_dir + str(id)
        if os.path.isfile(file_path):
            result = []
            facts = readFile(file_path).split('\n')

            for fact in facts:
                if fact:
                    hex = fact.split('\t')
                    assert(len(hex) == 6)

                    result.append(
                        ((hex[0], hex[1]),
                         hex[2],
                         (hex[3], hex[4]))
                    )

        else:
            return []
