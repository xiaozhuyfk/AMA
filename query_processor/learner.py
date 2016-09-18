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
    SELECT ?name where {
        ?x fb:type.object.name ?o .
        FILTER (lcase(str(?o)) = "albert einstein") .
        ?x fb:common.topic.alias ?name .
        FILTER (lang(?name) = "en")
    }
    '''

    query = '''
        PREFIX fb: <http://rdf.freebase.com/ns/>
        SELECT ?s ?name where {
            ?s fb:common.topic.alias ?o .
            FILTER (lcase(str(?o)) = "%s") .
            ?s fb:type.object.name ?name
        }
        ''' % "the professor"

    get_name = '''
        PREFIX fb: <http://rdf.freebase.com/ns/>
        SELECT ?o where {
            fb:%s fb:type.object.name ?o
        }
    '''
    #for x in backend.query(query):
    #    s, s_name = x[0], x[1]
    #    print s, s_name

    #for i in modules.extractor.extract_fact_list_with_str("the professor"):
    #    print i
    #for q in queries:
        #print q.id, q.utterance

    for query in queries:
        modules.extractor.extract_fact_list_with_entity_linker(query)


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