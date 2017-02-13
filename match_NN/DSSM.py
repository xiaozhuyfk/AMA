import numpy as np
from traitlets.config import Configurable
from traitlets import (
    Int,
    Float,
    Bool,
    Unicode,
)
import sys
reload(sys)
sys.setdefaultencoding('UTF8')
sys.path.append("..")
sys.path.append(".")


import time
import itertools
import argparse
from traitlets.config.loader import PyFileConfigLoader
from match_NN.base import ClickNN
from keras import backend
from keras.layers import Input, merge
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.convolutional import Convolution1D
from keras.models import Model
from keras.models import load_model






class DSSM(ClickNN):
    char_size = Int(6000, help='number of total characters').tag(config=True)
    window_size = Int(3, help='window size').tag(config=True)

    def __init__(self, **kwargs):
        super(DSSM, self).__init__(**kwargs)
        self.K = 300  # Dimensionality of the max-pooling layer. See section 3.4.
        self.L = 128  # Dimensionality of latent semantic space. See section 3.5.
        self.J = 4  # Number of random unclicked documents serving as negative examples for a query. See section 4.
        self.FILTER_LENGTH = 1  # We only consider one time step for convolutions.

    def build_model(self):
        """
        The model graph
        :return: return the score predicted for each document in the batch
        """
        # Input tensors holding the query, positive (clicked) document, and negative (unclicked) documents.
        # The first dimension is None because the queries and documents can vary in length.
        query = Input(shape=(None, self.char_size * self.window_size), name="query")
        pos_doc = Input(shape=(None, self.char_size * self.window_size), name="pos_doc")
        neg_docs = [Input(shape=(None, self.char_size * self.window_size), name="neg_doc_{0}".format(j)) for j in range(self.J)]

        # Query model. The paper uses separate neural nets for queries and documents (see section 5.2).

        # In this step, we transform each word vector with WORD_DEPTH dimensions into its
        # convolved representation with K dimensions. K is the number of kernels/filters
        # being used in the operation. Essentially, the operation is taking the dot product
        # of a single weight matrix (W_c) with each of the word vectors (l_t) from the
        # query matrix (l_Q), adding a bias vector (b_c), and then applying the tanh function.
        # That is, h_Q = tanh(W_c l_Q + b_c). With that being said, that's not actuall
        # how the operation is being calculated here. To tie the weights of the weight
        # matrix (W_c) together, we have to use a one-dimensional convolutional layer.
        # Further, we have to transpose our query matrix (l_Q) so that time is the first
        # dimension rather than the second (as described in the paper). That is, l_Q[0, :]
        # represents our first word vector rather than l_Q[:, 0]. We can think of the weight
        # matrix (W_c) as being similarly transposed such that each kernel is a column
        # of W_c. Therefore, h_Q = tanh(l_Q W_c + b_c) with l_Q, W_c, and b_c being
        # the transposes of the matrices described in the paper.

        # equation (2).
        query_conv = Convolution1D(self.K, self.FILTER_LENGTH, border_mode="same", input_shape=(None, self.char_size * self.window_size),
                                   activation="tanh", name="query_conv")(query)

        # Next, we apply a max-pooling layer to the convolved query matrix. Keras provides
        # its own max-pooling layers, but they cannot handle variable length input (as
        # far as I can tell). As a result, I define my own max-pooling layer here. In the
        # paper, the operation selects the maximum value for each row of h_Q, but, because
        # we're using the transpose, we're selecting the maximum value for each column.
        query_max = Lambda(lambda x: backend.max(x, axis=1), output_shape=(self.K,), name="query_max")(query_conv)

        # In this step, we generate the semantic vector represenation of the query. This
        # is a standard neural network dense layer, i.e., y = tanh(W_s v + b_s).
        query_sem = Dense(self.L, activation="tanh", input_dim=self.K, name='query_sem')(query_max)  # See section 3.5.

        # The document equivalent of the above query model.
        doc_conv = Convolution1D(self.K, self.FILTER_LENGTH, border_mode="same", input_shape=(None, self.char_size * self.window_size),
                                 activation="tanh")
        doc_max = Lambda(lambda x: backend.max(x, axis=1), output_shape=(self.K,))
        doc_sem = Dense(self.L, activation="tanh", input_dim=self.K)

        pos_doc_conv = doc_conv(pos_doc)
        neg_doc_convs = [doc_conv(neg_doc) for neg_doc in neg_docs]

        pos_doc_max = doc_max(pos_doc_conv)
        neg_doc_maxes = [doc_max(neg_doc_conv) for neg_doc_conv in neg_doc_convs]

        pos_doc_sem = doc_sem(pos_doc_max)
        neg_doc_sems = [doc_sem(neg_doc_max) for neg_doc_max in neg_doc_maxes]

        # This layer calculates the cosine similarity between the semantic representations of
        # a query and a document.
        query_sem = Reshape((1, 128))(query_sem)
        pos_doc_sem = Reshape((1, 128))(pos_doc_sem)
        R_Q_D_p = merge([query_sem, pos_doc_sem], mode="cos", dot_axes=2)  # See equation (4).

        R_Q_D_ns = [merge([query_sem, Reshape((1, 128))(neg_doc_sem)], mode="cos", dot_axes=2) for neg_doc_sem in neg_doc_sems]  # See equation (4).

        concat_Rs = merge([R_Q_D_p] + R_Q_D_ns, mode="concat")
        concat_Rs = Reshape((self.J + 1, 1))(concat_Rs)

        # In this step, we multiply each R(Q, D) value by gamma. In the paper, gamma is
        # described as a smoothing factor for the softmax function, and it's set empirically
        # on a held-out data set. We're going to learn gamma's value by pretending it's
        # a single 1 x 1 kernel.
        weight = np.array([1]).reshape(1, 1, 1, 1)
        with_gamma = Convolution1D(1, 1, border_mode="same", input_shape=(self.J + 1, 1), activation="linear", bias=False,
                                   weights=[weight])(concat_Rs)  # See equation (5).

        # Next, we exponentiate each of the gamma x R(Q, D) values.
        exponentiated = Lambda(lambda x: backend.exp(x), output_shape=(self.J + 1,))(with_gamma)  # See equation (5).
        exponentiated = Reshape((self.J + 1,))(exponentiated)

        # Finally, we use the softmax function to calculate the P(D+|Q).
        prob = Lambda(lambda x: backend.softmax(x), output_shape=(5,))(exponentiated)

        # We now have everything we need to define our model.
        model = Model(input=[query, pos_doc] + neg_docs, output=prob)
        model.compile(optimizer="adadelta", loss="categorical_crossentropy")
        get_R_Q_D_p = backend.function([query, pos_doc], [R_Q_D_p])

        return model, get_R_Q_D_p

        # Build a random data set.
        # sample_size = 10
        # l_Qs = []
        # pos_l_Ds = []

        # for i in range(sample_size):
        #   query_len = np.random.randint(1, 10)
        #    l_Q = np.random.rand(1, query_len, self.char_size)
        #    l_Qs.append(l_Q)

        #    doc_len = np.random.randint(50, 500)
        #    l_D = np.random.rand(1, doc_len, self.char_size)
        #    pos_l_Ds.append(l_D)

        # neg_l_Ds = []
        # for i in range(sample_size):
        #    possibilities = list(range(sample_size))
        #    possibilities.remove(i)
        #    negatives = np.random.choice(possibilities, self.J)
        #    neg_l_Ds.append([pos_l_Ds[negative] for negative in negatives])



        # for i in range(sample_size):
        #    history = model.fit([l_Qs[i], pos_l_Ds[i]] + neg_l_Ds[i], y, nb_epoch=1, verbose=0)

        # Here, I walk through how to define a function for calculating output from the
        # computational graph. Let's define a function that calculates R(Q, D+) for a given
        # query and clicked document. The function depends on two inputs, query and pos_doc.
        # That is, if you start at the point in the graph where R(Q, D+) is calculated
        # and then work backwards as far as possible, you'll end up at two different starting
        # points: query and pos_doc. As a result, we supply those inputs in a list to the
        # function. This particular function only calculates a single output, but multiple
        # outputs are possible (see the next example).
        # get_R_Q_D_p = backend.function([query, pos_doc], R_Q_D_p)
        # get_R_Q_D_p([l_Qs[0], pos_l_Ds[0]])

    def train_and_test(self, train_pair_file_path, train_size, test_file_path, checkpoint_dir, load_model=False):

        model, get_R_Q_D_p = self.build_model()
        start_time = time.time()

        # Because we're using the "categorical_crossentropy" loss function, we can pretend that
        # we're dealing with a multi-class classification problem and that every sample is a
        # member of the "0" class.
        y = np.zeros((self.batch_size, self.J + 1))
        for b in range(self.batch_size):
            y[b][0] = 1

        # Loop through training steps.
        train_stream = itertools.cycle(open(train_pair_file_path))

        for step in xrange(self.max_epochs * train_size / self.batch_size):
            X = next(self.generate_train_data(train_stream=train_stream, batch_size=self.batch_size))
            history = model.train_on_batch([X[u'q'], X[u'q']] + X[u'aux_d'], y)

            if step % self.eval_frequency == 0:
                # output evaluations
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('Step %d (epoch %.2f), %.1f ms per step' % (step,
                                                             float(step) * self.batch_size / (train_size),
                                                             1000 * elapsed_time / self.eval_frequency))
                print history
                sys.stdout.flush()

            if step % self.checkpoint_steps == 0:
                model.save(checkpoint_dir + '/model')
                print "model saved to " + checkpoint_dir
                sys.stdout.flush()

            # testing...
            if step > 0 and step % 500 == 0:
                test_stream = open(test_file_path)
                fout = open(checkpoint_dir + '/test_{0}.out'.format(step), 'w')
                for t, X in enumerate(self.generate_test_data(test_stream=test_stream)):
                    if t % 500 == 0:
                        print "testing {0} samples...".format(t)
                    rel = get_R_Q_D_p([X[u'q'], X[u'd']])
                    rel = rel[0]
                    fout.write(str(rel[0][0][0][0]))
                    fout.write('\n')
                fout.close()
                print "test result write to {0}/test_{1}.out".format(checkpoint_dir, step)

        del model

    def window_word_hash(self, text_str, max_len):
        """
        Hash a text into a char_size * window_size vector
        :param text_str:
        :return: (None, char_size * window_size)
        """
        tokens = [t for t in text_str.split(' ') if t]
        tokens = ['0'] + tokens + ['0']
        res = []
        for i in range(max_len): # window = tokens[i, i + 1, i + 2]
            vec = [0] * (self.char_size * self.window_size)
            if i + self.window_size - 1 >= len(tokens):
                res.append(vec)
                continue
            for j in range(self.window_size):
                token = tokens[i + j]
                chars = [int(c) for c in token.strip().split('_') if c]
                for c in chars:
                    vec[self.char_size * j + c] += 1
            res.append(vec)
        return res

    def generate_train_data(self, train_stream, batch_size):
        """
        Read training instances
        :param train_stream:
             each line: c1_c2 c3_c4... \t c1_c2 c3_c4.. \t...
        :return:
            X[u'q']: (None, char_size * window_size)
            X[u'd']: (None, char_size * window_size)
            X[u'aux_d']: [(None, char_size * window_size)] * J
        """

        while True:
            batch_q = []
            batch_pd = []
            batch_nd = [[] for j in range(self.J)]
            for b in range(batch_size):
                line = train_stream.next()
                items = line.split('\t')
                if len(items) != 6:
                    continue
                q_str = items[0]
                p_d_str = items[1]
                n_d_strs = items[2:6]
                q_vec = self.window_word_hash(q_str, 8)
                p_d_vec = self.window_word_hash(p_d_str, 32)

                batch_q.append(q_vec)
                batch_pd.append(p_d_vec)

                for j, n_d_str in enumerate(n_d_strs):
                    n_d_vec = self.window_word_hash(n_d_str, 32)
                    batch_nd[j].append(n_d_vec)

            for j in range(self.J):
                batch_nd[j] = np.array(batch_nd[j])

            res = {u'q': np.reshape(batch_q, (self.batch_size, len(batch_q[0]), self.char_size * self.window_size)),
                   u'd': np.array(batch_pd),
                   u'aux_d': batch_nd}
            yield res

    def generate_test_data(self, test_stream):
        """
        each time yield 1 test sample
        :param test_stream:
        :return:
        """
        for line in test_stream:
            items = line.strip().split('\t')
            if len(items) != 2:
                continue
            q_str = items[0]
            d_str = items[1]
            q_vec = self.window_word_hash(q_str, 8)
            d_vec = self.window_word_hash(d_str, 32)

            res = {u'q': np.array([q_vec]), u'd': np.array([d_vec])}
            yield res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file_path")

    parser.add_argument("--train", action='store_true')
    parser.add_argument("--train_file", '-f', help="train_pair_file_path")
    parser.add_argument("--train_size", '-z', type=int, help="number of train samples")
    parser.add_argument("--load_model", '-l', action='store_true')

    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_file")
    parser.add_argument("--test_size", type=int)
    parser.add_argument("--output_score_file", '-o')
    parser.add_argument("--emb_file_path", '-e')

    parser.add_argument("--checkpoint_dir", '-s', help="store model from here")

    args = parser.parse_args()

    conf = PyFileConfigLoader(args.config_file_path).load_config()

    if args.train:
        nn = DSSM(config=conf)
        nn.train_and_test(train_pair_file_path=args.train_file,
                          train_size=args.train_size,
                          test_file_path=args.test_file,
                          checkpoint_dir=args.checkpoint_dir,
                          load_model=args.load_model)
    else:
        raise NotImplementedError
        #nn.test(test_point_file_path=args.test_file,
        #        test_size=args.test_size,
        #        output_file_path=args.output_score_file,
        #        load_model=True,
        #        emb_file_path=args.emb_file_path,
        #        checkpoint_dir=args.checkpoint_dir)
