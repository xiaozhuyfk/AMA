import logging
import os
import modules
from util import readFile, codecsWriteFile

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

        self.facts_by_id_query = '''
        PREFIX fb: <http://rdf.freebase.com/ns/>
        SELECT ?r ?o where {
            fb:%s ?r ?o
        }
        '''

    @staticmethod
    def init_from_config(args, config_options):
        fact_list_dir = config_options.get('FactList',
                                        args.dataset)
        return FactExtractor(fact_list_dir)

    def store_fact_list(self, query, fact_list):
        id = query.id
        file_path = self.fact_list_dir + str(id)
        codecsWriteFile(file_path, query.utterance + "\n")
        for fact in fact_list:
            line = "\t".join(fact) + "\n"
            codecsWriteFile(file_path, line, "a")


    def extract_fact_list_with_entity_linker(self, query):
        logger.info("Extracting facts with entity linker from question: " + query.utterance)

        if self.fact_list_on_disk(query):
            return self.load_fact_list_from_disk(query)
        else:
            question = query.utterance.lower()
            parse_result = modules.parser.parse(question)
            tokens = parse_result.tokens

            entities = modules.entity_linker.identify_entities_in_tokens(tokens)
            result = []
            for ie in entities:
                e = ie.entity
                s, s_name = e.id, e.name
                facts = self.backend.query(self.facts_by_id_query % s)
                for f in facts:
                    r, o = f[0], f[1]
                    if o.startswith('m.'):
                        o_name = self.backend.query(self.name_by_id_query % o)
                        # skip if the entity does not have a name in Freebase
                        if o_name == []:
                            continue
                        hex = (s, s_name, r, o, o_name[0][0])
                        result.append(hex)
                    else:
                        hex = (s, s_name, r, "ATTRIBUTE", o)
                        result.append(hex)
            self.store_fact_list(query, result)
            return result

    def extract_fact_list_with_ngram(self, query):
        logger.info("Extracting facts from question: " + query.utterance)

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
            return self.extract_fact_list_with_str(token)



    def extract_fact_list_with_str(self, q):
        entities = self.backend.query(self.entities_with_alias_query % q)
        result = []
        for x in entities:
            s, s_name = x[0], x[1]
            facts = self.backend.query(self.facts_by_id_query % s)
            for f in facts:
                r, o = f[0], f[1]
                if o.startswith('m.'):
                    o_name = self.backend.query(self.name_by_id_query % o)
                    # skip if the entity does not have a name in Freebase
                    if o_name == []:
                        continue
                    hex = (s, s_name, r, o, o_name)
                    result.append(hex)
                else:
                    hex = (s, s_name, r, "ATTRIBUTE", o)
                    result.append(hex)
        return result

    def fact_list_on_disk(self, query):
        id = query.id
        file_path = self.fact_list_dir + str(id)
        return os.path.isfile(file_path)

    def load_fact_list_from_disk(self, query):
        id = query.id
        file_path = self.fact_list_dir + str(id)
        if os.path.isfile(file_path):
            result = []
            facts = readFile(file_path).strip().split('\n')[1:]

            for fact in facts:
                if fact:
                    hex = fact.split('\t')

                    result.append(
                        (hex[0], hex[1],
                         hex[2],
                         hex[3], hex[4])
                    )
            return result

        else:
            return []
