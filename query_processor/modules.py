import logging
import globals
from word2vec import Word2Vec

logger = logging.getLogger(__name__)

w2v = None
sparql_backend = None

def init_from_config():
    global w2v, sparql_backend
    config_options = globals.config

    embeddings_model = config_options.get('Alignment',
                                          'word-embeddings')
    w2v = Word2Vec(embeddings_model)
    sparql_backend = globals.get_sparql_backend(config_options)