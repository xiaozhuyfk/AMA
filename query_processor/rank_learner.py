from __future__ import print_function

import logging
import globals
import modules
from util import codecsDumpJson

logging.basicConfig(format='%(asctime)s : %(levelname)s '
                           ': %(module)s : %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def train_model(model_name):
    modules.facts_ranker.train_model(model_name)

def train(dataset):
    modules.facts_ranker.train(dataset)

def test(dataset):
    modules.facts_ranker.test(dataset)

def extract(dataset):
    train_data = modules.facts_ranker.extract_fact_candidates("webquestionstrain")
    test_data = modules.facts_ranker.extract_fact_candidates("webquestionstest")

    vocab_train = train_data.get("vocab")
    vocab_train_trigram = train_data.get("vocab_trigram")
    vocab_test = test_data.get("vocab")
    vocab_test_trigram = test_data.get("vocab_trigram")

    print(len(vocab_train))
    print(len(vocab_test))

    vocab = sorted(vocab_train | vocab_test)
    vocab_trigram = sorted(vocab_train_trigram | vocab_test_trigram)

    codecsDumpJson("/home/hongyul/AMA/training_model/vocab-rank", vocab)
    codecsDumpJson("/home/hongyul/AMA/training_model/vocab-rank-trigram", vocab_trigram)

    print(train_data.get("sentence_size"))
    print(test_data.get("sentence_size"))
    print(train_data.get("sentence_trigram_size"))
    print(test_data.get("sentence_trigram_size"))



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

    process_parser = subparsers.add_parser('model', help="Train model")
    process_parser.add_argument('dataset',
                                help='Training dataset')
    process_parser.add_argument('name',
                                help='Training model name')
    process_parser.set_defaults(which='model')

    extract_parser = subparsers.add_parser('extract', help="Extract fact candidates")
    extract_parser.add_argument('dataset',
                                help='Training dataset')
    extract_parser.set_defaults(which='extract')

    args = parser.parse_args()

    # Read global config
    globals.read_configuration(args.config)

    # Load modules
    modules.init_from_config(args)

    if args.which == 'train':
        train(args.dataset)
    elif args.which == 'test':
        test(args.dataset)
    elif args.which == 'model':
        train_model(args.name)
    elif args.which == 'extract':
        extract(args.dataset)


if __name__ == '__main__':
    main()