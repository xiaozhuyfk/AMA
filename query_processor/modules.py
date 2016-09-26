import logging
import globals
from word2vec import Word2Vec
from fact_extractor import FactExtractor
from corenlp_parser.parser import CoreNLPParser
from entity_linker.entity_linker import EntityLinker

logger = logging.getLogger(__name__)

w2v = None
sparql_backend = None
extractor = None
parser = None
entity_linker = None

def init_from_config(args):
    global w2v, sparql_backend, extractor, parser, entity_linker
    config_options = globals.config

    w2v = Word2Vec.init_from_config(config_options)
    #sparql_backend = globals.get_sparql_backend(config_options)
    #extractor = FactExtractor.init_from_config(args, config_options)
    parser = CoreNLPParser.init_from_config()
    #entity_linker = EntityLinker.init_from_config()
