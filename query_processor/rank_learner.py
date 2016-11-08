from __future__ import print_function

import logging
import globals
import modules
from evaluation import load_eval_queries
from util import codecsWriteFile, codecsReadFile, codecsLoadJson, kstem, codecsDumpJson
import codecs
import re
import numpy as np
import random
import json
import os
import tensorflow as tf
from memory_network import MemN2N
from data_utils import vectorize_data, memory_data, selective_data
from itertools import chain
from sklearn import cross_validation, metrics

logging.basicConfig(format='%(asctime)s : %(levelname)s '
                           ': %(module)s : %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


stems = {}

def load_data_from_disk(query, path):
        id = query.id
        file_path = path + str(id)
        if os.path.isfile(file_path):
            d = codecsLoadJson(file_path)
            return d
        else:
            return None

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

def process_data(dataset, path):
    queries = load_eval_queries(dataset)
    sentence_size = 0
    vocab = set([])

    for query in queries:
        logger.info("Processing question " + str(query.id))

        d = load_data_from_disk(query, path)
        if d is not None:
            q = d.get("query")
            s = d.get("story")
            a = d.get("answer")
            for ss in s:
                if len(ss) > sentence_size:
                    sentence_size = len(ss)
            if len(q) > sentence_size:
                sentence_size = len(q)
            vocab |= set(list(chain.from_iterable(s)) + q + a)
            continue

    return vocab, sentence_size

    """
        data_path = path + str(query.id)
        codecsWriteFile(data_path, "")

        facts = modules.extractor.extract_fact_list_with_entity_linker(query)
        question = query.utterance.lower()[:-1]
        tokens = [tokenize_term(t) for t in question.split()]
        story = []
        S = []
        R = []
        O = []
        y = []
        sstory = []
        SS = []
        RR = []
        OO = []
        yy = []
        answer = query.target_result

        story_set = set([])
        for fact in facts:
            sid, s, r, oid, o = fact
            if (o.startswith("g.")):
                continue
            relations = re.split("\.\.|\.", r)[-2:]
            rels = [tokenize_term(e) for t in relations for e in re.split('\.\.|\.|_', t)]
            subjects = [re.sub('[?!@#$%^&*,()_+=\'/]', '', t).lower() for t in s.split()]
            objects = [re.sub('[?!@#$%^&*,()_+=\']', '', t).lower() for t in o.split()]
            if (len(objects) > 10):
                continue
            sentence = subjects + rels #+ objects
            story.append(sentence)
            S.append(subjects)
            R.append(rels)
            O.append(objects)
            y.append((o in answer) * 1.0)

            s = " ".join(sentence)
            if s not in story_set:
                if len(sentence) > sentence_size:
                    sentence_size = len(sentence)
                sstory.append(sentence)
                story_set.add(s)

                SS.append(subjects)
                RR.append(rels)
                OO.append(objects)
                yy.append((o in answer) * 1.0)

        d = {"query" : tokens,
             "story" : story,
             "answer" : answer,
             "S": S,
             "R": R,
             "O": O,
             "y": y,
             "sstory" : sstory,
             "SS" : SS,
             "RR" : RR,
             "OO" : OO,
             "yy" : yy}

        if len(tokens) > sentence_size:
            sentence_size = len(tokens)

        with codecs.open(data_path, mode='w', encoding='utf-8') as f:
            json.dump(d, f, indent=4)
    logger.info("Longest sentence size: " + str(sentence_size))
    """


tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 1, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 30, "Number of epochs to trai`n for.")
tf.flags.DEFINE_integer("embedding_size", 20, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 1, "bAbI task id, 1 <= id <= 20")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "data/tasks_1-20_v1-2/en/", "Directory containing bAbI tasks")
FLAGS = tf.flags.FLAGS

def load_data(dataset):
    """
    config_options = globals.config
    vocab_file = config_options.get('Train', 'vocab')
    training_data = config_options.get('Train', 'training-data')
    testing_data = config_options.get('Test', 'testing-data')

    vocab1, size1 = process_data("webquestionstrain", training_data)
    vocab2, size2 = process_data("webquestionstest", testing_data)
    vocab = sorted(vocab1 | vocab2)
    codecsDumpJson(vocab_file, vocab)

    print(size1)
    print(size2)
    """

    queries = load_eval_queries(dataset)
    for query in queries:
        logger.info("Processing question " + str(query.id))
        modules.extractor.extract_fact_list_with_entity_linker(query)
    logger.info("Done extracting new fact list.")


def training_progress_message(epoch, epochs, query_id, total, loss):
    progress = ("Progress: %.2f" % (float(query_id + 1) / total * 100)) + "%"
    message = "Processing question " + str(query_id) + ". "
    training = "Epoch %d/%d: loss = %f. " % (epoch, epochs, loss)

    return message + training + progress + " "*10 + "\r"

def randomize_input(trainS, trainQ, trainA):
    shuffleS = []
    shuffleQ = []
    shuffleA = []
    indices = range(len(trainS))
    random.shuffle(indices)

    for i in indices:
        shuffleS.append(trainS[i])
        shuffleQ.append(trainQ[i])
        shuffleA.append(trainA[i])

    trainS = np.array(shuffleS)
    trainQ = np.array(shuffleQ)
    trainA = np.array(shuffleA)
    return trainS, trainQ, trainA

def train(dataset):
    """
    load_data(dataset)
    exit()

    config_options = globals.config
    vocab_file = config_options.get('Train', 'vocab')
    training_data = config_options.get('Train', 'training-data')
    model_file = config_options.get('Train', 'model-file')

    vocab = codecsLoadJson(vocab_file)
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    """
    modules.facts_ranker.train(dataset)





def test(dataset):
    config_options = globals.config
    vocab_file = config_options.get('Train', 'vocab')
    testing_data = config_options.get('Test', 'testing-data')
    model_file = config_options.get('Train', 'model-file')
    test_result = config_options.get('Test', 'test-result')

    vocab = codecsLoadJson(vocab_file)
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    batch_size = FLAGS.batch_size
    sentence_size = 17
    memory_size = 10
    vocab_size = len(word_idx) + 1

    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate,
                                       epsilon=FLAGS.epsilon)

    with tf.Session() as sess:
        model = MemN2N(batch_size,
                       vocab_size,
                       sentence_size,
                       memory_size,
                       64,
                       session=sess,
                       hops=FLAGS.hops,
                       max_grad_norm=FLAGS.max_grad_norm,
                       optimizer=optimizer)
        model.load_model(model_file)

        queries = load_eval_queries(dataset)
        data = []
        for i in xrange(len(queries)):
            query = queries[i]
            d = load_data_from_disk(query, testing_data)
            data.append(d)
        S, Q, A = selective_data(data, word_idx, sentence_size, memory_size)

        test_labels = np.argmax(A, axis=1)
        test_preds = model.predict(S, Q)
        test_acc = metrics.accuracy_score(test_preds, test_labels)
        print("Testing Accuracy:", test_acc)

        aqqu_labels = np.zeros(len(test_labels))
        aqqu_acc = metrics.accuracy_score(aqqu_labels, test_labels)
        print("Aqqu Accuracy:", aqqu_acc)

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
        process_trainingdata(args.dataset)


if __name__ == '__main__':
    main()