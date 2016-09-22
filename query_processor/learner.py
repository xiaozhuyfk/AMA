import logging
import globals
import modules
from evaluation import load_eval_queries
from util import codecsWriteFile, codecsReadFile



logging.basicConfig(format='%(asctime)s : %(levelname)s '
                           ': %(module)s : %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def train(dataset):
    """
    queries = load_eval_queries(dataset)
    codecsWriteFile("trainingdata", "")
    for query in queries:
        facts = modules.extractor.extract_fact_list_with_entity_linker(query)

        question = query.utterance.lower()
        logger.info("Processing question " + question)
        parse_result = modules.parser.parse(question)
        tokens = parse_result.tokens

        answer = query.target_result
        for fact in facts:
            sid, s, r, oid, o = fact
            if (o in answer):
                line = question + "\t" + "\t".join(fact) + "\t" + "1" + "\n"
                codecsWriteFile("trainingdata", line, 'a')
            else:
                line = question + "\t" + "\t".join(fact) + "\t" + "0" + "\n"
                codecsWriteFile("trainingdata", line, 'a')
    """
    data = codecsReadFile("trainingdata").strip().split("\n")
    print modules.w2v.w2v("Hongyu")




def test(dataset):
    pass


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

    args = parser.parse_args()

    # Read global config
    globals.read_configuration(args.config)

    # Load modules
    modules.init_from_config(args)

    if args.which == 'train':
        train(args.dataset)
    elif args.which == 'test':
        test(args.dataset)


if __name__ == '__main__':
    main()