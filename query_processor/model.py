import logging
import random
import numpy as np
from util import codecsLoadJson
from keras.layers import (
    Input,
    LSTM,
    Dense,
    Embedding,
    Merge,
    Bidirectional,
    Dropout,
    merge,
    Convolution1D,
    GlobalMaxPooling1D,
)
from keras.models import (
    Model,
    model_from_json,
    Sequential,
)

logger = logging.getLogger(__name__)


def vectorize_sentence(word_idx, sentence, sentence_size):
    sentence_idx = [word_idx.get(t, 0) for t in sentence] + \
                   (sentence_size - len(sentence)) * [0]
    return sentence_idx

class BaseModel(object):

    def __init__(self, config_options, name):
        self.config_options = config_options
        self.name = name

        self.vocab_dim = int(config_options.get(name, 'vocab-dim'))
        vocab_path = config_options.get(name, 'vocab')
        self.vocab = codecsLoadJson(vocab_path)
        self.word_idx = dict((c, i + 1) for i, c in enumerate(self.vocab))
        self.n_symbols = len(self.vocab) + 1

        self.model = None
        self.ranking_model = None
        self.model_struct = config_options.get(name, 'model-struct')
        self.model_weights = config_options.get(name, 'model-weights')

        self.nb_epoch = int(config_options.get(name, 'nb-epoch'))
        self.batch_size = int(config_options.get(name, 'batch-size'))
        self.sentence_size = int(config_options.get(name, 'sentence-size'))
        self.sentence_attr = config_options.get(name, 'sentence-attr')


class CNNPairwise(BaseModel):

    def predict(self, query_candidates, sentence_size):
        self.load_model()
        x = []
        for candidate in query_candidates:
            sentence = getattr(candidate, self.sentence_attr)
            sentence_idx = vectorize_sentence(self.word_idx, sentence, sentence_size)
            x.append(sentence_idx)
        return self.ranking_model.predict(np.array(x))

    def load_model(self):
        name = self.name
        self.model_struct = self.config_options.get(name, 'model-struct')
        self.model_weights = self.config_options.get(name, 'model-weights')
        self.ranking_model_struct = self.config_options.get(name, 'ranking-model-struct')
        self.ranking_model_weights = self.config_options.get(name, 'ranking-model-weights')

        if self.model is None:
            self.model = load_model(self.model_struct,
                                    self.model_weights)
        if self.ranking_model is None:
            self.ranking_model = load_model(self.ranking_model_struct,
                                            self.ranking_model_weights)


    def save_model(self):
        name = self.name
        self.model_struct = self.config_options.get(name, 'model-struct')
        self.model_weights = self.config_options.get(name, 'model-weights')
        self.ranking_model_struct = self.config_options.get(name, 'ranking-model-struct')
        self.ranking_model_weights = self.config_options.get(name, 'ranking-model-weights')
        save_model_to_file(self.model,
                           self.model_struct,
                           self.model_weights)
        save_model_to_file(self.ranking_model,
                           self.ranking_model_struct,
                           self.ranking_model_weights)


    def train(self, candidates, sentence_size, query_attr, fact_attr):
        self.model, self.ranking_model = self._build_model(
            vocab_dim=self.vocab_dim,
            n_symbols=self.n_symbols,
            word_idx=self.word_idx,
        )

        X, Y = self._construct_data(candidates,
                                    sentence_size,
                                    query_attr,
                                    fact_attr)
        self.model.fit(X, Y,
                       batch_size=self.batch_size,
                       nb_epoch=self.nb_epoch)
        self.save_model()


    def _construct_data(self, candidates, sentence_size, query_attr, fact_attr):
        pool1 = []
        pool2 = []
        label = []
        for query_candidates in candidates:
            correct = []
            wrong = []
            for candidate in query_candidates:
                if candidate.f1 > 0:
                    correct.append(candidate)
                else:
                    wrong.append(candidate)
            wrong = random.sample(wrong, min(len(wrong), 20))
            for i in xrange(len(correct)):
                for j in xrange(len(wrong)):
                    if random.randint(0,1):
                        pool1.append(correct[i])
                        pool2.append(wrong[j])
                        label.append(1.0)
                    else:
                        pool1.append(wrong[j])
                        pool2.append(correct[i])
                        label.append(-1.0)

        Q1 = []
        F1 = []
        Q2 = []
        F2 = []
        for i in xrange(len(label)):
            q1 = getattr(pool1[i], query_attr)
            f1 = getattr(pool1[i], fact_attr)
            q2 = getattr(pool2[i], query_attr)
            f2 = getattr(pool2[i], fact_attr)
            q1_idx = vectorize_sentence(self.word_idx, q1, sentence_size)
            f1_idx = vectorize_sentence(self.word_idx, f1, sentence_size)
            q2_idx = vectorize_sentence(self.word_idx, q2, sentence_size)
            f2_idx = vectorize_sentence(self.word_idx, f2, sentence_size)
            Q1.append(q1_idx)
            F1.append(f1_idx)
            Q2.append(q2_idx)
            F2.append(f2_idx)

        return [np.array(Q1), np.array(F1), np.array(Q2), np.array(F2)], np.array(label)

    def _build_model(self, vocab_dim, n_symbols, word_idx):
        logger.info("Constructing CNN model.")
        embedding = Embedding(output_dim=vocab_dim,
                              input_dim=n_symbols,
                              name='embedding',
                              dropout=0.2)
        cnn = Convolution1D(nb_filter=300,
                            filter_length=3,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1)
        pooling = GlobalMaxPooling1D()
        dense = Dense(100)

        l_question_input = Input(shape=(self.sentence_size,))
        l_fact_input = Input(shape=(self.sentence_size,))
        l_question = embedding(l_question_input)
        l_question = cnn(l_question)
        l_question = pooling(l_question)
        l_question = dense(l_question)
        l_fact = embedding(l_fact_input)
        l_fact = cnn(l_fact)
        l_fact = pooling(l_fact)
        l_fact = dense(l_fact)
        l_merged = merge([l_question, l_fact],
                       mode='cos',
                       output_shape=(1,))

        r_question_input = Input(shape=(self.sentence_size,))
        r_fact_input = Input(shape=(self.sentence_size,))
        r_question = embedding(r_question_input)
        r_question = cnn(r_question)
        r_question = pooling(r_question)
        r_question = dense(r_question)
        r_fact = embedding(r_fact_input)
        r_fact = cnn(r_fact)
        r_fact = pooling(r_fact)
        r_fact = dense(r_fact)
        r_merged = merge([r_question, r_fact],
                       mode='cos',
                       output_shape=(1,))

        ranking_model = Model(input=[l_question_input, l_fact_input], output=l_merged)

        merged = merge([l_merged, r_merged],
                       mode=lambda x: x[0] - x[1],
                       output_shape=(1,))
        model = Model(input=[l_question_input, l_fact_input,
                             r_question_input, r_fact_input],
                      output=merged)
        model.compile(optimizer='rmsprop',
                      loss='hinge',
                      metrics=['accuracy'])

        return model, ranking_model


class LSTMPairwise(BaseModel):

    def _candidate_compare(self, sentence_size):
        def compare(item1, item2):
            s1 = np.array([vectorize_sentence(self.word_idx,
                                             getattr(item1, self.sentence_attr),
                                             sentence_size)])
            s2 = np.array([vectorize_sentence(self.word_idx,
                                             getattr(item2, self.sentence_attr),
                                             sentence_size)])
            result = self.model.predict([s1, s2])[0][0]
            return (result > 0) * 1 + (result < 0) * -1
        return compare


    def predict(self, query_candidates, sentence_size):
        self.load_model()
        x = []
        for candidate in query_candidates:
            sentence = getattr(candidate, self.sentence_attr)
            sentence_idx = vectorize_sentence(self.word_idx, sentence, sentence_size)
            x.append(sentence_idx)
        return self.ranking_model.predict(np.array(x))

    def load_model(self):
        name = self.name
        self.model_struct = self.config_options.get(name, 'model-struct')
        self.model_weights = self.config_options.get(name, 'model-weights')
        self.ranking_model_struct = self.config_options.get(name, 'ranking-model-struct')
        self.ranking_model_weights = self.config_options.get(name, 'ranking-model-weights')

        if self.model is None:
            self.model = load_model(self.model_struct,
                                    self.model_weights)
        if self.ranking_model is None:
            self.ranking_model = load_model(self.ranking_model_struct,
                                            self.ranking_model_weights)


    def save_model(self):
        name = self.name
        self.model_struct = self.config_options.get(name, 'model-struct')
        self.model_weights = self.config_options.get(name, 'model-weights')
        self.ranking_model_struct = self.config_options.get(name, 'ranking-model-struct')
        self.ranking_model_weights = self.config_options.get(name, 'ranking-model-weights')
        save_model_to_file(self.model,
                           self.model_struct,
                           self.model_weights)
        save_model_to_file(self.ranking_model,
                           self.ranking_model_struct,
                           self.ranking_model_weights)


    def train(self, candidates, sentence_size):
        self.model, self.ranking_model = self._build_model(
            vocab_dim=self.vocab_dim,
            n_symbols=self.n_symbols,
            word_idx=self.word_idx,
        )

        X, Y = self._construct_data(candidates, sentence_size)
        self.model.fit(X, Y,
                       batch_size=self.batch_size,
                       nb_epoch=self.nb_epoch)
        self.save_model()


    def _construct_data(self, candidates, sentence_size):
        pool1 = []
        pool2 = []
        label = []
        for query_candidates in candidates:
            correct = []
            wrong = []
            for candidate in query_candidates:
                if candidate.f1 > 0:
                    correct.append(candidate)
                else:
                    wrong.append(candidate)
            #wrong = random.sample(wrong, min(len(wrong), 20))
            for i in xrange(len(correct)):
                for j in xrange(len(wrong)):
                    if random.randint(0,1):
                        pool1.append(correct[i])
                        pool2.append(wrong[j])
                        label.append(1.0)
                    else:
                        pool1.append(wrong[j])
                        pool2.append(correct[i])
                        label.append(-1.0)

        X1 = []
        X2 = []
        for i in xrange(len(label)):
            sentence1 = getattr(pool1[i], self.sentence_attr)
            sentence2 = getattr(pool2[i], self.sentence_attr)
            sentence1_idx = vectorize_sentence(self.word_idx, sentence1, sentence_size)
            sentence2_idx = vectorize_sentence(self.word_idx, sentence2, sentence_size)
            X1.append(sentence1_idx)
            X2.append(sentence2_idx)

        return [np.array(X1), np.array(X2)], np.array(label)

    def _build_model(self, vocab_dim, n_symbols, word_idx):
        logger.info("Constructing LSTM Pairwise model.")
        left_input = Input(shape=(self.sentence_size,))
        right_input = Input(shape=(self.sentence_size,))
        embedding = Embedding(output_dim=vocab_dim,
                              input_dim=n_symbols,
                              mask_zero=True,
                              name='embedding')
        lstm = Bidirectional(LSTM(16, name='lstm'))
        dense = Dense(1, name='dense')

        left = embedding(left_input)
        left = Dropout(0.2)(left)
        left = lstm(left)
        left = Dropout(0.2)(left)
        left = dense(left)
        left_model = Model(input=left_input, output=left)

        right = embedding(right_input)
        right = Dropout(0.2)(right)
        right = lstm(right)
        right = Dropout(0.2)(right)
        right = dense(right)
        right_model = Model(input=right_input, output=right)

        merged = merge([left, right],
                       mode=lambda x: x[0] - x[1],
                       output_shape=(1,))

        model = Model(input=[left_input, right_input], output=merged)
        model.compile(optimizer='rmsprop',
                      loss='hinge',
                      metrics=['accuracy'])

        return model, left_model



class LSTMPointwise(BaseModel):

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
            sentence = getattr(candidate, self.sentence_attr)
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
            sentence = getattr(candidate, self.sentence_attr)
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