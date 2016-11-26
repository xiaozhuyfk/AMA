import logging
import globals
import random
import numpy as np
from util import codecsLoadJson
from keras.layers import Input, LSTM, Dense, Embedding, Merge, Bidirectional, Dropout
from keras.models import Model, model_from_json, Sequential

logger = logging.getLogger(__name__)


def vectorize_sentence(word_idx, sentence, sentence_size):
    sentence_idx = [word_idx.get(t, 0) for t in sentence] + \
                   (sentence_size - len(sentence)) * [0]
    return sentence_idx


class LSTMPointwise(object):

    def __init__(self, config_options, name):
        self.config_options = config_options
        self.name = name

        self.vocab_dim = int(config_options.get(name, 'vocab-dim'))
        vocab_path = config_options.get(name, 'vocab')
        self.vocab = codecsLoadJson(vocab_path)
        self.word_idx = dict((c, i + 1) for i, c in enumerate(self.vocab))
        self.n_symbols = len(self.vocab) + 1

        self.model = None
        self.model_struct = config_options.get(name, 'model-struct')
        self.model_weights = config_options.get(name, 'model-weights')

        self.nb_epoch = int(config_options.get(name, 'nb-epoch'))
        self.batch_size = int(config_options.get(name, 'batch-size'))

    def train(self, candidates, sentence_size):
        self.model = self._build_model(
            vocab_dim=self.vocab_dim,
            n_symbols=self.n_symbols,
            word_idx=self.word_idx,
        )

        X, Y = self._construct_data(candidates, sentence_size)
        self.model.fit(X, Y,
                       batch_size=self.batch_size,
                       nb_epoch=self.nb_epoch)
        save_model_to_file(self.model, self.model_struct, self.model_weights)


    def predict(self, query_candidates, sentence_size):
        if self.model is None:
            self.model = load_model(self.model_struct, self.model_weights)
        x = []
        for candidate in query_candidates:
            sentence = candidate.sentence
            sentence_idx = vectorize_sentence(self.word_idx, sentence, sentence_size)
            x.append(sentence_idx)
        return self.model.predict(np.array(x))

    def _construct_data(self, candidates, sentence_size):
        X = []
        Y = []
        positive = []
        negative = []
        for query_candidates in candidates:
            wrong = []
            for candidate in query_candidates:
                if candidate.f1 > 0:
                    positive.append(candidate)
                else:
                    wrong.append(candidate)
            negative += random.sample(wrong, min(len(wrong), 10))

        pool = positive + negative
        random.shuffle(pool)

        for candidate in pool:
            sentence = candidate.sentence
            sentence_idx = vectorize_sentence(self.word_idx, sentence, sentence_size)
            X.append(sentence_idx)
            Y.append(candidate.relevance)
        return np.array(X), np.array(Y)

    def _build_model(self, vocab_dim, n_symbols, word_idx):
        logger.info("Constructing Bi-directional LSTM model.")
        model = Sequential()

        model.add(
            Embedding(output_dim=vocab_dim,
                      input_dim=n_symbols,
                      mask_zero=True)
        )
        model.add(Dropout(0.2))

        model.add(
            Bidirectional(LSTM(16))
        )
        model.add(Dropout(0.2))

        model.add(
            Dense(1, activation="sigmoid")
        )
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=["accuracy"])

        return model

class LSTMPointwiseTrigram(LSTMPointwise):

    def _construct_data(self, candidates, sentence_size):
        X = []
        Y = []
        positive = []
        negative = []
        for query_candidates in candidates:
            wrong = []
            for candidate in query_candidates:
                if candidate.f1 > 0:
                    positive.append(candidate)
                else:
                    wrong.append(candidate)
            negative += random.sample(wrong, min(len(wrong), 10))

        pool = positive + negative
        random.shuffle(pool)

        for candidate in pool:
            sentence = candidate.sentence_trigram
            sentence_idx = vectorize_sentence(sentence, self.word_idx, sentence_size)
            X.append(sentence_idx)
            Y.append(candidate.f1)
        return np.array(X), np.array(Y)

    def predict(self, query_candidates, sentence_size):
        if self.model is None:
            self.model = load_model(self.model_struct, self.model_weights)
        x = []
        for candidate in query_candidates:
            sentence = candidate.sentence_trigram
            sentence_idx = vectorize_sentence(self.word_idx, sentence, sentence_size)
            x.append(sentence_idx)
        return self.model.predict(np.array(x))


def save_model_to_file(model, struct_file, weights_file):
    # save model structure
    model_struct = model.to_json()
    open(struct_file, 'w').write(model_struct)

    # save model weights
    model.save_weights(weights_file, overwrite=True)


def load_model(struct_file, weights_file):
    model = model_from_json(open(struct_file, 'r').read())
    model.load_weights(weights_file)
    return model