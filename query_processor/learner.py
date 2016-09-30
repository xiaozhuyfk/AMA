from __future__ import print_function

import logging
import globals
import modules
from evaluation import load_eval_queries
from util import codecsWriteFile, codecsReadFile
import codecs
import re
from keras.layers import Input, LSTM, Dense
from keras.models import Model, model_from_json, Sequential
import numpy as np
import random

logging.basicConfig(format='%(asctime)s : %(levelname)s '
                           ': %(module)s : %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def save_model_to_file(model, struct_file, weights_file):
    # save model structure
    model_struct = model.to_json()
    open(struct_file, 'w').write(model_struct)

    # save model weights
    model.save_weights(weights_file, overwrite=True)


def load_model(struct_file, weights_file):
    model = model_from_json(open(struct_file, 'r').read())
    model.compile(loss="categorical_crossentropy", optimizer='sgd')
    model.load_weights(weights_file)

    return model


def transform_to_vectors(tokens, input_dim):
    vectors = np.zeros((input_dim, 300))
    valid = []
    for word in tokens:
        word = re.sub('[?!@#$%^&*,()_+=]', '', word)
        v = modules.w2v.transform(word)
        if v is not None:
            valid.append(v)
    for i in xrange(len(valid)):
        vectors[i] = valid[i]
    return vectors


def process_line(line, input_dim):
    words = line.strip().split()
    label = float(words[-1])
    vectors = transform_to_vectors(words[:-1], input_dim)
    return vectors, label

def generate_data_from_file(path, input_dim):
    f = codecs.open(path, mode="rt", encoding="utf-8")
    while True:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            x, y = process_line(line, input_dim)
            yield (np.array([x]), np.array([y]))
    f.close()



def process_trainingdata(dataset):
    queries = load_eval_queries(dataset)
    codecsWriteFile("training_pos.dat", "")
    count = 0
    for query in queries:
        facts = modules.extractor.extract_fact_list_with_entity_linker(query)

        question = query.utterance.lower()[:-1]
        logger.info("Processing question " + str(query.id))
        hasAnwer = False

        answer = query.target_result
        correct = []
        wrong = []
        for fact in facts:
            sid, s, r, oid, o = fact
            if (o.startswith("g.")):
                continue
            if (o in answer):
                #line = question + "\t" + "\t".join(fact) + "\t" + "1" + "\n"
                #codecsWriteFile("trainingdata", line, 'a')
                correct.append(fact)
                hasAnwer = True
            else:
                #line = question + "\t" + "\t".join(fact) + "\t" + "0" + "\n"
                #codecsWriteFile("trainingdata", line, 'a')
                wrong.append(fact)

        if not hasAnwer:
            logger.info(question + " does not have an answer.")
            count += 1

        for fact in correct:
            sid, s, r, oid, o = fact
            relations = re.split("\.\.|\.", r)[:-2]
            rels = [e for t in relations for e in re.split('\.\.|\.|_', t)]

            tokens = [re.sub('[?!@#$%^&*,()_+=\']', '', t) for t in question.split()]
            subjects = [re.sub('[?!@#$%^&*,()_+=\']', '', t) for t in s.split()]
            objects = [re.sub('[?!@#$%^&*,()_+=\']', '', t) for t in o.split()]

            line = "\t".join(tokens + subjects + rels + objects + ["1.0"]) + "\n"
            codecsWriteFile("training_pos.dat", line, "a")

        sample = wrong
        if len(correct) == 0:
            if len(wrong) > 5:
                sample = random.sample(wrong, 5)
        elif len(correct) * 5 > len(wrong):
            sample = wrong
        else:
            sample = random.sample(wrong, len(correct) * 5)

        for fact in sample:
            sid, s, r, oid, o = fact
            relations = re.split("\.\.|\.", r)[:-2]
            rels = [e for t in relations for e in re.split('\.\.|\.|_', t)]

            tokens = [re.sub('[?!@#$%^&*,()_+=\']', '', t) for t in question.split()]
            subjects = [re.sub('[?!@#$%^&*,()_+=\']', '', t) for t in s.split()]
            objects = [re.sub('[?!@#$%^&*,()_+=\']', '', t) for t in o.split()]

            line = "\t".join(tokens + subjects + rels + objects + ["0.0"]) + "\n"
            codecsWriteFile("training_pos.dat", line, "a")

    logger.info(str(count) + " questions do not have answers.")


def train(dataset):

    #process_trainingdata(dataset)

    print(modules.w2v.transform("Paris"))

    """
    f = codecs.open("training.dat", mode="rt", encoding="utf-8")
    for line in f:
        line = line.strip()
        if line == "":
            continue
        x, y = process_line(line, 518)
        print x
        print y
    """

    """
    f = codecs.open("trainingdata", mode="rt", encoding="utf-8")
    codecsWriteFile("training_pos.dat", "")
    longest = 0

    curr = ""
    count = 0
    for line in f:
        if line == "":
            continue
        data = line.split("\t")
        query = data[0].lower()
        s = data[2].lower()
        r = data[3].lower()
        o = data[5].lower()
        label = data[6]

        if o.startswith("g."):
            continue

        if query != curr:
            curr = query
            count += 1
            logger.info("Processing question count: " + str(count))


        #tokens = [t.token for t in modules.parser.parse(query).tokens]
        #relations = re.split('\.\.|\.|_', r)
        #subjects = [t.token for t in modules.parser.parse(s).tokens]
        #objects = [t.token for t in modules.parser.parse(o).tokens]

        tokens = query[:-1].split()
        relations = re.split('\.\.|\.|_', r)
        subjects = s.split()
        objects = o.split()

        result = tokens + subjects + relations + objects + [label]
        if (len(result) > longest):
            longest = len(result)
        codecsWriteFile("training.dat", "\t".join(result) + "\n", 'a')

    logger.info("Longest sequence is " + str(longest))
    f.close()
    """

    """
    model = Sequential()
    model.add(LSTM(32, input_shape=(None, 300)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    f = codecs.open("training.dat", mode="rt", encoding="utf-8")
    X = []
    Y = []

    for line in f:
        line = line.strip()
        if line == "":
            continue
        x, y = process_line(line, 518)
        X.append(x)
        Y.append(y)

        if (len(X) >= 32):
            X = np.array(X)
            Y = np.array(Y)
            model.fit(X, Y)

            X = []
            Y = []

    if X != []:
        X = np.array(X)
        Y = np.array(Y)
        model.fit(X, Y)

    f.close()
    """

    #model.fit_generator(generate_data_from_file('training.dat', length),
    #                    samples_per_epoch=100,
    #                    nb_epoch=10)

    #model.fit(X, Y)

    #save_model_to_file(model, "modelstruct", "modelweights")


def test(dataset):
    model = load_model("modelstruct", "modelweights")
    input_dim = 518

    queries = load_eval_queries(dataset)
    codecsWriteFile("result.txt", "")
    for query in queries:
        facts = modules.extractor.extract_fact_list_with_entity_linker(query)

        question = query.utterance.lower()
        logger.info("Testing question " + question)
        #tokens = [t.token for t in modules.parser.parse(question).tokens]
        tokens = question[:-1].split()
        answer = query.target_result

        input_facts = []
        for fact in facts:
            sid, s, r, oid, o = fact
            if not o.startswith("g."):
                input_facts.append(fact)

        inputs = []
        for fact in input_facts:
            sid, s, r, oid, o = fact

            relations = re.split('\.\.|\.|_', r)
            #subjects = [t.token for t in modules.parser.parse(s).tokens]
            #objects = [t.token for t in modules.parser.parse(o).tokens]
            subjects = s.split()
            objects = o.split()

            sentence = tokens + subjects + relations + objects
            input_vector = transform_to_vectors(sentence, input_dim)
            inputs.append(input_vector)

        inputs = np.array(inputs)
        scores = model.predict(inputs)

        predictions = []
        print(scores)
        for i in xrange(len(scores)):
            score = scores[i]
            sid, s, r, oid, o = input_facts[i]
            if score == 1.0:
                predictions.append(o)

        result_line = "\t".join([question, str(answer), str(predictions)]) + "\n"
        codecsWriteFile("result.txt", result_line, "a")


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