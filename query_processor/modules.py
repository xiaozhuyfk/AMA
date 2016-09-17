import logging
import globals
from word2vec import Word2Vec

logger = logging.getLogger(__name__)

w2v = None

def init_from_config():
    global w2v
    config_options = globals.config

    embeddings_model = config_options.get('Alignment',
                                          'word-embeddings')
    w2v = Word2Vec(embeddings_model)