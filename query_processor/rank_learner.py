from __future__ import print_function

import logging
import globals
import modules
from evaluation import load_eval_queries
from util import codecsWriteFile, codecsReadFile, codecsLoadJson, kstem, codecsDumpJson
import re

logging.basicConfig(format='%(asctime)s : %(levelname)s '
                           ': %(module)s : %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

def tokenize_term(t):
    return re.sub('[?!@#$%^&*,()_+=\'\d\./]', '', t).lower()

def get_stem(t):
    global stems
    if t in stems:
        return stems[t]
    else:
        stem = kstem(t)
        stems[t] = stem
        return stem

def train(dataset):
    #modules.facts_ranker.train_model()
    modules.facts_ranker.train(dataset)

def test(dataset):
    modules.facts_ranker.test(dataset)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Choose to learn or test AMA')

    parser.add_argument('--config',
                        default='config.cfg',
                        help='The configuration file to use')
    subparsers = parser.add_subparsers(help='command help')
    train_parser = subparsers.add_parser('train', help='Train memory network')
    train_parser.add_argument('dataset',
                              help='The dataset to train.')
    train_parser.set_defaults(which='train')

    test_parser = subparsers.add_parser('test', help='Test memory network')
    test_parser.add_argument('dataset',
                             help='The dataset to test')
    test_parser.set_defaults(which='test')

    process_parser = subparsers.add_parser('process', help="Process training data")
    process_parser.add_argument('dataset',
                                help='Training data file')
    process_parser.set_defaults(which='process')

    args = parser.parse_args()

    # Read global config
    globals.read_configuration(args.config)

    # Load modules
    modules.init_from_config(args)

    if args.which == 'train':
        train(args.dataset)
    elif args.which == 'test':
        test(args.dataset)
    elif args.which == 'process':
        pass


if __name__ == '__main__':
    main()