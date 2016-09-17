import logging
import globals
import modules
from evaluation import load_eval_queries



logging.basicConfig(format='%(asctime)s : %(levelname)s '
                           ': %(module)s : %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def train(dataset):
    queries = load_eval_queries(dataset)
    backend = modules.sparql_backend
    query = '''
    PREFIX fb: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?x ?o
    WHERE {
     ?s fb:type.object.name "Albert Einstein"@EN .
     ?s ?p ?o .
     FILTER regex(?p, "profession") .
     ?o fb:type.object.name ?x .
     FILTER (LANG(?x) = "en") }
    '''

    query = '''
    PREFIX fb: <http://rdf.freebase.com/ns/>
    SELECT ?x where {
        ?x fb:type.object.alias "Albert Einstein"@EN .
        ?x ?r ?name .
        FILTER (lang(?name) = "en")
    }
    '''
    print backend.query(query)

    #for q in queries:
        #print q.id, q.utterance



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