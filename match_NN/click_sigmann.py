import tensorflow as tf
import numpy as np
from deeplearning4ir.model import ClickDataGenerator
from traitlets.config import Configurable
from traitlets import (
    Int,
    Float,
    Bool,
    Unicode,
)
import sys
import time
import itertools
import argparse
from traitlets.config.loader import PyFileConfigLoader
from gensim.models import Word2Vec
from deeplearning4ir.match_NN import ClickNN

reload(sys)
sys.setdefaultencoding('UTF8')


class ClickSigmaNN(ClickNN):
    emb_in = Unicode(help="embedding in").tag(config=True)
    emb_binary = Bool(False, help="whether the embedding is binary file").tag(config=True)

    def __init__(self, **kwargs):
        super(ClickSigmaNN, self).__init__(**kwargs)

        # init mus
        self.mus = tf.constant(ClickNN.kernal_mus(self.n_bins + 1, use_exact=True)[1:], shape=[self.n_bins])
        self.mus = tf.reshape(self.mus, [1, 1, self.n_bins])

        # The variables below hold all the trainable weights. They are passed an
        # initial value which will be assigned when we call:
        # {tf.initialize_all_variables().run()}
        self.W1 = ClickNN.weight_variable([self.n_bins + 1, 5])
        self.b1 = tf.Variable(tf.zeros([5]))
        self.W2 = ClickNN.weight_variable([5, 1])
        self.b2 = tf.Variable(tf.zeros([1]))
        self.sigma = tf.constant([1.0/self.n_bins] * self.n_bins, shape=[self.n_bins])
        self.sigma = tf.reshape(self.sigma, [1, 1, self.n_bins])

        # load embeddings
        print "loading word2vec..."
        # self.emb = Word2Vec.load_word2vec_format(self.emb_in, unicode_errors='ignore', binary=self.emb_binary)
        self.emb = self.load_word2vec(self.emb_in)

    def load_word2vec(self, emb_file_path):
        emb = {}
        nlines = 0
        with open(emb_file_path) as f:
            for line in f:
                nlines += 1
                if nlines == 1:
                    continue
                items = line.split()
                tid = int(items[0])
                if tid >= self.vocabulary_size:
                    print tid
                    continue
                vec = np.array([float(t) for t in items[1:]])
                emb[tid] = vec
                if nlines % 100000 == 0:
                    print "load {0} vectors...".format(nlines)
                    # break

        return emb

    def fetch_embeddings(self, X):
        res = np.ones((X.shape[0], X.shape[1], self.embedding_size))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if X[i][j] < 0:
                    break
                if X[i][j] in self.emb:
                    vec = self.emb[X[i][j]]
                    res[i, j, :] = vec
                else:
                    X[i][j] = 0
        return res

    def calculate_cosine_similarity(self, X):
        sim_mat_batches = []
        sim_mat_aux_batches = []
        train_size = len(X['q'])
        for b in xrange(train_size / self.batch_size):
            if b % 100 == 1:
                print b
            q_embed = self.fetch_embeddings(X['q'][b * self.batch_size: min((b + 1) * self.batch_size, train_size)])
            d_embed = self.fetch_embeddings(X['d'][b * self.batch_size: min((b + 1) * self.batch_size, train_size)])

            norm_q = np.sqrt(np.sum(np.square(q_embed), 2, keepdims=True))
            normalized_q_embed = q_embed / norm_q

            norm_d = np.sqrt(np.sum(np.square(d_embed), 2, keepdims=True))
            normalized_d_embed = d_embed / norm_d
            tmp1 = np.transpose(normalized_d_embed, axes=[0, 2, 1])

            sim_mat = np.matmul(normalized_q_embed, tmp1)
            sim_mat_batches.append(sim_mat)

            if u'd_aux' in X and X[u'd_aux'] is not None and len(X[u'd_aux']) > 0:
                d_aux_embed = self.fetch_embeddings(X['d_aux'][b * self.batch_size: min((b + 1) * self.batch_size, train_size)])

                norm_d_aux = np.sqrt(np.sum(np.square(d_aux_embed), 2, keepdims=True))
                normalized_d_aux_embed = d_aux_embed / norm_d_aux  # [batch_size, max_d_len, emb_size]
                tmp2 = np.transpose(normalized_d_aux_embed, axes=[0, 2, 1])

                sim_mat_aux = np.matmul(normalized_q_embed, tmp2)
                sim_mat_aux_batches.append(sim_mat_aux)

        return sim_mat_batches, sim_mat_aux_batches

    def model(self, sim, doc_mask, idf):
        """
        The pointwise model graph
        :return: return the score predicted for each document in the batch
        """

        # compute gaussian kernel
        rs_sim = tf.reshape(sim, [self.batch_size, self.max_q_len, self.max_d_len, 1])

        # ignore exact match

        soft = tf.exp(-tf.square(tf.sub(rs_sim, self.mus)) / (tf.mul(tf.square(self.sigma), 2))) # [batch_size, mqx_q, max_d, n_bins]
        exact = tf.exp(-tf.square(tf.sub(rs_sim, 1.0)) / (tf.mul(tf.square(0.000000001), 2)))
        tmp = tf.concat(3, (exact, soft))
        kde = tf.reduce_sum(tmp * doc_mask, [2])
        kde = tf.log(tf.maximum(kde, 1e-10)) * 0.01  # scale the data
        print kde.get_shape()

        # aggregated by idf
        aggregated_kde = tf.reduce_sum(kde * idf, [1])

        # dense NN built upon the KDE
        h1 = tf.tanh(tf.matmul(aggregated_kde, self.W1) + self.b1)
        o = tf.tanh(tf.matmul(h1, self.W2) + self.b2)

        return aggregated_kde, o

    def train(self, train_pair_file_path, val_pair_file_path, train_size, checkpoint_dir, load_module=False):

        # PLACEHOLDERS
        # This is where training samples and labels are fed to the graph.
        # These placeholder nodes will be fed a batch of training data at each
        # training step using the {feed_dict} argument to the Run() call below.

        # nodes to hold query and qterm idf. padding terms will have idf=0
        train_input_idf = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_q_len], name='idf')

        # nodes to hold posive sample similarity matrix
        pos_sim = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_q_len, self.max_d_len])

        # nodes to hold negative samples
        neg_sim = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_q_len, self.max_d_len])

        # mask padding terms
        # assume all docid >= 1
        # padding with 0
        # also mask out exact match
        input_train_mask_pos = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_d_len])
        input_train_mask_neg = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_d_len])

        # reshape place holders
        rs_train_doc_mask_pos = tf.reshape(input_train_mask_pos, [self.batch_size, 1, self.max_d_len, 1])
        rs_train_doc_mask_neg = tf.reshape(input_train_mask_neg, [self.batch_size, 1, self.max_d_len, 1])
        rs_idf = tf.reshape(train_input_idf, shape=[self.batch_size, self.max_q_len, 1])

        # training graph
        akde_pos, o_pos = self.model(pos_sim, rs_train_doc_mask_pos, rs_idf)
        akde_neg, o_neg = self.model(neg_sim, rs_train_doc_mask_neg, rs_idf)
        loss = tf.reduce_mean(tf.maximum(0.0, 1 - o_pos + o_neg))

        # optimizer
        lr = 0.001
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        # Create a local session to run the training.

        with tf.Session() as sess:
            pair_stream = itertools.cycle(open(train_pair_file_path))
            val_pair_stream = itertools.cycle(open(val_pair_file_path))

            saver = tf.train.Saver()
            start_time = time.time()

            # Run all the initializers to prepare the trainable parameters.
            if not load_module:
                print "Initialize variables..."
                tf.initialize_all_variables().run()
                print('Initialized!')
                step_offset = 0

            else:
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    #step_offset = int(ckpt.model_checkpoint_path.strip().split('-')[-1])
                    print "model loaded!"
                else:
                    print "no model found"
                    exit(-1)


            print ("Calculating similarities...")
            X_all, Y_all = next(self.data_generator.pairwise_reader(pair_stream, train_size, with_idf=True))
            pos_sim_batches, neg_sim_batches = self.calculate_cosine_similarity(X_all)
            X_val_all, Y_val_all = next(self.val_data_generator.pairwise_reader(val_pair_stream, 10000, with_idf=True))
            val_pos_sim_batches, val_neg_sim_batches = self.calculate_cosine_similarity(X_val_all)
            print ("Finish calculating similarities!")

            for step in xrange(self.max_epochs * train_size / self.batch_size):
                b = step % (train_size / self.batch_size)

                X_d = X_all[u'd'][b * self.batch_size: min((b + 1) * self.batch_size, train_size)]
                X_aux_d = X_all[u'd_aux'][b * self.batch_size: min((b + 1) * self.batch_size, train_size)]
                X_idf = X_all[u'idf'][b * self.batch_size: min((b + 1) * self.batch_size, train_size)]

                pos_sim_mat = pos_sim_batches[b]
                neg_sim_mat = neg_sim_batches[b]

                if len(X_idf) != self.batch_size:
                    continue

                M_pos = np.zeros((self.batch_size, self.max_d_len))
                M_neg = np.zeros((self.batch_size, self.max_d_len))
                M_pos[X_d > 0] = 1
                M_neg[X_aux_d > 0] = 1

                train_feed_dict = {pos_sim: pos_sim_mat,
                                   neg_sim: neg_sim_mat,
                                   train_input_idf: self.re_pad(X_idf, self.batch_size),
                                   input_train_mask_pos: M_pos,
                                   input_train_mask_neg: M_neg}

                # Run the graph and fetch some of the nodes.
                _, l, ap, an = sess.run([optimizer, loss, akde_pos, akde_neg], feed_dict=train_feed_dict)


                val_l = 0
                if step % self.eval_frequency == 0:

                    # compute validation loss
                    if step % self.eval_frequency == 0:
                        val_l = 0
                        for b in range(10000 / self.batch_size):
                            X_d = X_val_all[u'd'][b * self.batch_size: min((b + 1) * self.batch_size, 10000)]
                            X_aux_d = X_val_all[u'd_aux'][b * self.batch_size: min((b + 1) * self.batch_size, 10000)]
                            X_idf = X_val_all[u'idf'][b * self.batch_size: min((b + 1) * self.batch_size, 10000)]
                            pos_sim_mat = val_pos_sim_batches[b]
                            neg_sim_mat = val_neg_sim_batches[b]

                            M_pos = np.zeros((self.batch_size, self.max_d_len))
                            M_neg = np.zeros((self.batch_size, self.max_d_len))
                            M_pos[X_d > 0] = 1
                            M_neg[X_aux_d > 0] = 1

                            val_feed_dict = {pos_sim: pos_sim_mat,
                                             neg_sim: neg_sim_mat,
                                             train_input_idf: self.re_pad(X_idf, self.batch_size),
                                             input_train_mask_pos: M_pos,
                                             input_train_mask_neg: M_neg}
                            val_l += sess.run(loss, feed_dict=val_feed_dict)
                        val_l /= 10000 / self.batch_size

                    # output evaluations
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print('Step %d (epoch %.2f), %.1f ms per step' % (step,
                                                             float(step) * self.batch_size / train_size,
                                                             1000 * elapsed_time / self.eval_frequency))
                    print('Minibatch loss: %.3f, validation loss: %.3f' % (l, val_l))
                    print sess.run(self.sigma)
                    print sess.run(self.mus)
                    sys.stdout.flush()

                # save model
                if (step + 1) % self.checkpoint_steps == 0:
                    saver.save(sess, checkpoint_dir + '/model.ckpt')
                    print sess.run(self.sigma)
                    print sess.run(self.mus)
            saver.save(sess, checkpoint_dir + '/model.ckpt')

    def test(self, test_point_file_path, test_size, output_file_path, checkpoint_dir=None, load_model=False):

        # PLACEHOLDERS
        # This is where training samples and labels are fed to the graph.
        # These placeholder nodes will be fed a batch of training data at each
        # training step using the {feed_dict} argument to the Run() call below.

        # nodes to hold mu sigma
        input_mu = tf.placeholder(tf.float32, shape=[self.n_bins], name='input_mu')

        # nodes to hold query and qterm idf. padding terms will have idf=0
        test_input_idf = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_q_len], name='idf')

        # nodes to hold posive sample similarity matrix
        test_sim = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_q_len, self.max_d_len])

        # mask padding terms
        # assume all docid >= 1
        # assume padded with 0
        test_mask = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_d_len])

        # reshape place holders
        mu = tf.reshape(input_mu, shape=[1, 1, self.n_bins])
        rs_test_mask = tf.reshape(test_mask, [self.batch_size, 1, self.max_d_len, 1])
        rs_idf = tf.reshape(test_input_idf, shape=[self.batch_size, self.max_q_len, 1])

        # training graph
        akde, o = self.model(test_sim, rs_test_mask, rs_idf)

        # Create a local session to run the testing.

        with tf.Session() as sess:
            test_point_stream = open(test_point_file_path)
            outfile = open(output_file_path, 'w')
            saver = tf.train.Saver()

            if load_model:
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print "model loaded!"
                else:
                    print "no model found"
                    exit(-1)

            # Loop through training steps.
            print sess.run(self.W1)
            print sess.run(self.W2)
            for b in range(int(np.ceil(float(test_size)/self.batch_size))):
                X, Y = next(self.test_data_generator.pointwise_generate(test_point_stream, self.batch_size, with_idf=True, with_label=False))
                X[u'q'] = self.re_pad(X[u'q'], self.batch_size)
                X[u'd'] = self.re_pad(X[u'd'], self.batch_size)
                M = np.zeros((self.batch_size, self.max_d_len))
                M[X[u'd'] > 0] = 1
                sim_mat, _ = self.calculate_cosine_similarity(X)
                if len(sim_mat) > 1:
                    print sim_mat
                    exit()
                if len(sim_mat) < 1:
                    exit()
                test_feed_dict = {test_sim: sim_mat[0],
                                  test_input_idf: self.re_pad(X[u'idf'], self.batch_size),
                                  test_mask: M}

                # Run the graph and fetch some of the nodes.
                scores, a = sess.run([o, akde], feed_dict=test_feed_dict)

                #for t in a:
                #    outfile.write(' '.join([str(ti) for ti in t]))
                #    outfile.write('\n')
                for score in scores:
                    outfile.write('{0}\n'.format(score[0]))

            outfile.close()
            test_point_stream.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file_path")

    parser.add_argument("--train", action='store_true')
    parser.add_argument("--train_file", '-f', help="train_pair_file_path")
    parser.add_argument("--validation_file", '-v', help="val_pair_file_path")
    parser.add_argument("--train_size", '-z', type=int, help="number of train samples")
    parser.add_argument("--load_module", '-l', action='store_true')

    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_file")
    parser.add_argument("--test_size", type=int)
    parser.add_argument("--output_score_file", '-o')
    parser.add_argument("--checkpoint_dir", '-s', help="store model from here")

    args = parser.parse_args()

    conf = PyFileConfigLoader(args.config_file_path).load_config()

    nn = None
    if args.train:
        nn = ClickSigmaNN(config=conf)
        nn.train(train_pair_file_path=args.train_file,
                 val_pair_file_path=args.validation_file,
                 train_size=args.train_size,
                 checkpoint_dir=args.checkpoint_dir,
                 load_module=args.load_module)
    if args.test:
        if not nn:
            nn = ClickSigmaNN(config=conf)
            nn.test(test_point_file_path=args.test_file,
                    test_size=args.test_size,
                    output_file_path=args.output_score_file,
                    load_model=True,
                    checkpoint_dir=args.checkpoint_dir)
        else:
            nn.test(test_point_file_path=args.test_file,
                    test_size=args.test_size,
                    output_file_path=args.output_score_file,
                    load_model=True,
                    checkpoint_dir=args.checkpoint_dir)