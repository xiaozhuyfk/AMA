import logging
import globals
from word2vec import Word2Vec
from fact_extractor import FactExtractor

logger = logging.getLogger(__name__)

w2v = None
sparql_backend = None
extractor = None

def init_from_config(args):
    global w2v, sparql_backend, extractor
    config_options = globals.config

    w2v = Word2Vec.init_from_config(config_options)
    sparql_backend = globals.get_sparql_backend(config_options)
    extractor = FactExtractor.init_from_config(args, config_options)