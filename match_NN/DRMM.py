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
from deeplearning4ir.match_NN import ClickNN


reload(sys)
sys.setdefaultencoding('UTF8')


class DRMM(ClickNN):
    neg_sample = Int(1, help='negative sample').tag(config=True)
    emb_in = Unicode('None', help="embedding in").tag(config=True)
    use_exact = True

    def __init__(self, **kwargs):
        super(DRMM, self).__init__(**kwargs)

        self.mus = DRMM.kernal_mus(self.n_bins, use_exact=True)

        self.emb = self.load_word2vec(self.emb_in)
        self.embeddings = tf.constant(self.emb, dtype='float32', shape=[self.vocabulary_size + 1, self.embedding_size])
        print "Initialized embeddings with {0}".format(self.emb_in)

        # The variables below hold all the trainable weights. They are passed an
        # initial value which will be assigned when we call:
        # {tf.initialize_all_variables().run()}
        self.W1 = DRMM.weight_variable([self.n_bins, 5])
        self.b1 = tf.Variable(tf.zeros([5]))
        self.W2 = DRMM.weight_variable([5, 1])
        self.b2 = tf.Variable(tf.zeros([1]))

    def load_word2vec(self, emb_file_path):
        emb = np.random.uniform(low=-1, high=1, size=(self.vocabulary_size + 1, self.embedding_size))
        nlines = 0
        with open(emb_file_path) as f:
            for line in f:
                nlines += 1
                if nlines == 1:
                    continue
                items = line.split()
                tid = int(items[0])
                if tid > self.vocabulary_size:
                    print tid
                    continue
                vec = np.array([float(t) for t in items[1:]])
                emb[tid, :] = vec
                if nlines % 20000 == 0:
                    print "load {0} vectors...".format(nlines)
        return emb

    def model(self, inputs_q, inputs_d, doc_mask, idf, mu):
        """
        The pointwise model graph
        :return: return the score predicted for each document in the batch
        """
        # look up embeddings for each term. [nbatch, qlen, emb_dim]
        q_embed = tf.nn.embedding_lookup(self.embeddings, inputs_q, name='qemb')
        d_embed = tf.nn.embedding_lookup(self.embeddings, inputs_d, name='demb')

        # normalize and compute similarity matrix
        norm_q = tf.sqrt(tf.reduce_sum(tf.square(q_embed), 2, keep_dims=True))
        normalized_q_embed = q_embed / norm_q
        norm_d = tf.sqrt(tf.reduce_sum(tf.square(d_embed), 2, keep_dims=True))
        normalized_d_embed = d_embed / norm_d
        tmp = tf.transpose(normalized_d_embed, perm=[0, 2, 1])

        # similarity matrix [n_batch, qlen, dlen]
        sim = tf.batch_matmul(normalized_q_embed, tmp, name='similarity_matrix')

        # compute histogram
        sims = tf.unpack(sim)  # batchsize * (q, d)
        doc_mask = tf.reshape(doc_mask, [self.batch_size, self.max_q_len, self.max_d_len])
        masks = tf.unpack(doc_mask) # batch * (q, d)
        q_hists = []

        # for every batch
        lala = None
        flag = False
        for sim, mask in zip(sims, masks):
            qtsims = tf.unpack(sim)  # q * (d)
            qmasks = tf.unpack(mask) # q * (d)
            hists = []

            # for every qterm
            for qtsim, qmask in zip(qtsims, qmasks):
                binned_values = tf.floor((qtsim + 1) / 2 * (self.n_bins - 1))  # d
                binned_values = tf.cast(binned_values, tf.int32)  # d

                ones = tf.ones_like(binned_values, dtype=tf.float32)  # d
                ones = ones * qmask  # d
                qd_hist = tf.unsorted_segment_sum(ones, binned_values,
                                               self.n_bins) # n_bins

                hists.append(qd_hist)

            q_hist = tf.concat(0, hists)  # q * n_nins
            q_hists.append(q_hist)

        hist1 = tf.concat(0, q_hists)  # batch * q * n_bins
        hist2 = tf.reshape(hist1, [self.batch_size * self.max_q_len, self.n_bins])
        hist3 = tf.log(tf.maximum(hist2, 1e-10)) * 0.01

        # feed-forward NN
        h1 = tf.tanh(tf.matmul(hist3, self.W1) + self.b1)  # (batch * q, 5)
        h2 = tf.tanh(tf.matmul(h1, self.W2) + self.b2)  # (batch * q, 1)

        # aggregated by idf
        h2 = tf.reshape(h2, [self.batch_size, self.max_q_len, 1]) # (batch, q, 1)
        o = tf.reduce_sum(h2 * idf, [1])  # (batch, 1)
        return o, hist2

    def train(self, train_pair_file_path, val_pair_file_path, train_size, checkpoint_dir, load_model=False):

        # PLACEHOLDERS
        # This is where training samples and labels are fed to the graph.
        # These placeholder nodes will be fed a batch of training data at each
        # training step using the {feed_dict} argument to the Run() call below.

        # nodes to hold mu sigma
        input_mu = tf.placeholder(tf.float32, shape=[self.n_bins], name='input_mu')

        # nodes to hold query and qterm idf. padding terms will have idf=0
        train_inputs_q = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_q_len], name='train_inputs_q')
        train_input_idf = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_q_len], name='idf')

        # nodes to hold training data, postive samples
        train_inputs_pos_d = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_d_len],
                                            name='train_inputs_pos_d')

        # nodes to hold negative samples
        train_inputs_neg_d = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_d_len])

        # mask padding terms
        # assume all docid >= 1
        # padding with 0
        # also mask out exact match
        input_train_mask_pos = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_q_len, self.max_d_len])
        input_train_mask_neg = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_q_len, self.max_d_len])

        # reshape place holders
        mu = tf.reshape(input_mu, shape=[1, 1, self.n_bins])
        rs_train_doc_mask_pos = tf.reshape(input_train_mask_pos, [self.batch_size, self.max_q_len, self.max_d_len, 1])
        rs_train_doc_mask_neg = tf.reshape(input_train_mask_neg, [self.batch_size, self.max_q_len, self.max_d_len, 1])
        rs_idf = tf.reshape(train_input_idf, shape=[self.batch_size, self.max_q_len, 1])

        # training graph
        o_pos, _ = self.model(train_inputs_q, train_inputs_pos_d, rs_train_doc_mask_pos, rs_idf, mu)
        o_neg, _ = self.model(train_inputs_q, train_inputs_neg_d, rs_train_doc_mask_neg, rs_idf, mu)
        loss = tf.reduce_mean(tf.maximum(0.0, 1 - o_pos + o_neg))

        # optimizer
        lr = 0.001
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

        # Create a local session to run the training.

        with tf.Session() as sess:
            pair_stream = itertools.cycle(open(train_pair_file_path))
            val_pair_stream = itertools.cycle(open(val_pair_file_path))

            saver = tf.train.Saver()
            start_time = time.time()

            # Run all the initializers to prepare the trainable parameters.
            if not load_model:
                print "Initialize variables..."
                tf.initialize_all_variables().run()
                print('Initialized!')

            else:
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print "model loaded!"
                else:
                    print "no model found"
                    exit(-1)

            # Loop through training steps.
            for step in xrange(self.max_epochs * train_size * self.neg_sample / self.batch_size):
                X, Y = next(self.data_generator.pairwise_reader(pair_stream, self.batch_size, with_idf=True))
                M_pos = self.gen_mask(X[u'q'], X[u'd'], self.use_exact)
                M_neg = self.gen_mask(X[u'q'], X[u'd_aux'], self.use_exact)
                if X[u'idf'].shape[0] != self.batch_size:
                    continue
                train_feed_dict = {train_inputs_q: self.re_pad(X[u'q'], self.batch_size),
                                   train_inputs_pos_d: self.re_pad(X[u'd'], self.batch_size),
                                   train_inputs_neg_d: self.re_pad(X[u'd_aux'], self.batch_size),
                                   train_input_idf: self.re_pad(X[u'idf'], self.batch_size),
                                   input_mu: self.mus,
                                   input_train_mask_pos: M_pos,
                                   input_train_mask_neg: M_neg}

                # Run the graph and fetch some of the nodes.
                _, l = sess.run([optimizer, loss], feed_dict=train_feed_dict)

                # eqp, edp, edn = sess.run([qp, dp, dn], feed_dict=train_feed_dict)
                # eb = sess.run(self.embeddings)

                val_l = 0
                if step > 0 and step % self.eval_frequency == 0:

                    # compute validation loss
                    if step % self.eval_frequency == 0:
                        val_l = 0
                        for b in range(50000 * self.neg_sample / self.batch_size):
                            X_val, Y_val = next(self.val_data_generator.pairwise_reader(val_pair_stream, self.batch_size, with_idf=True))
                            M_pos = self.gen_mask(X_val[u'q'], X_val[u'd'], self.use_exact)
                            M_neg = self.gen_mask(X_val[u'q'], X_val[u'd_aux'], self.use_exact)
                            val_feed_dict = {train_inputs_q: self.re_pad(X_val[u'q'], self.batch_size),
                                             train_inputs_pos_d: self.re_pad(X_val[u'd'], self.batch_size),
                                             train_inputs_neg_d: self.re_pad(X_val[u'd_aux'], self.batch_size),
                                             train_input_idf: self.re_pad(X_val[u'idf'], self.batch_size),
                                             input_mu: self.mus,
                                             input_train_mask_pos: M_pos,
                                             input_train_mask_neg: M_neg}
                            val_l += sess.run(loss, feed_dict=val_feed_dict)
                        val_l /= 50000 * self.neg_sample / self.batch_size

                    # output evaluations
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print('Step %d (epoch %.2f), %.1f ms per step' % (step,
                                                             float(step) * self.batch_size / (train_size * self.neg_sample),
                                                             1000 * elapsed_time / self.eval_frequency))
                    print('Minibatch loss: %.3f, validation loss: %.3f' % (l, val_l))
                    sys.stdout.flush()

                # save model
                if (step + 1) % self.checkpoint_steps == 0:
                    saver.save(sess, checkpoint_dir + '/model.ckpt')

            # end training
            #saver.save(sess, checkpoint_dir + '/model.ckpt')

    def test(self, test_point_file_path, test_size, output_file_path, emb_file_path, checkpoint_dir=None, load_model=False):

        # PLACEHOLDERS
        # This is where training samples and labels are fed to the graph.
        # These placeholder nodes will be fed a batch of training data at each
        # training step using the {feed_dict} argument to the Run() call below.

        # nodes to hold mu sigma
        input_mu = tf.placeholder(tf.float32, shape=[self.n_bins], name='input_mu')

        # nodes to hold query and qterm idf. padding terms will have idf=0
        test_inputs_q = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_q_len], name='test_inputs_q')
        test_input_idf = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_q_len], name='idf')

        # nodes to hold test data
        test_inputs_d = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_d_len],
                                       name='test_inputs_pos_d')

        # mask padding terms
        # assume all docid >= 1
        # assume padded with 0
        test_mask = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_q_len, self.max_d_len])

        # reshape place holders
        mu = tf.reshape(input_mu, shape=[1, 1, self.n_bins])
        rs_test_mask = tf.reshape(test_mask, [self.batch_size, self.max_q_len, self.max_d_len, 1])
        rs_idf = tf.reshape(test_input_idf, shape=[self.batch_size, self.max_q_len, 1])

        # training graph
        o = self.model(test_inputs_q, test_inputs_d, rs_test_mask, rs_idf, mu)

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

            emb = sess.run(self.embeddings)

            with open(emb_file_path, 'w') as fout:
                fout.write(str(self.vocabulary_size - 1) + ' ' + str(self.embedding_size))
                fout.write('\n')
                for t in range(1, self.vocabulary_size):
                    outstr = str(t) + ' '
                    outstr += ' '.join(["%.5f" % number for number in emb[t]])
                    fout.write(outstr)
                    fout.write('\n')
            print "Embedding file write to {0}".format(emb_file_path)

            # Loop through training steps.
            print sess.run(self.W1)
            print sess.run(self.W2)
            for b in range(int(np.ceil(float(test_size)/self.batch_size))):
                X, Y = next(self.test_data_generator.pointwise_generate(test_point_stream, self.batch_size, with_idf=True, with_label=False))
                M = self.gen_mask(X[u'q'], X[u'd'], use_exact=True)
                test_feed_dict = {test_inputs_q: self.re_pad(X[u'q'], self.batch_size),
                                  test_inputs_d: self.re_pad(X[u'd'], self.batch_size),
                                  test_input_idf: self.re_pad(X[u'idf'], self.batch_size),
                                  input_mu: self.mus,
                                  test_mask: M}

                # Run the graph and fetch some of the nodes.
                scores = sess.run(o, feed_dict=test_feed_dict)

                for score in scores:
                    outfile.write('{0}\n'.format(score))

            outfile.close()
            test_point_stream.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file_path")

    parser.add_argument("--train", action='store_true')
    parser.add_argument("--train_file", '-f', help="train_pair_file_path")
    parser.add_argument("--validation_file", '-v', help="val_pair_file_path")
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
        nn = DRMM(config=conf)
        nn.train(train_pair_file_path=args.train_file,
                 val_pair_file_path=args.validation_file,
                 train_size=args.train_size,
                 checkpoint_dir=args.checkpoint_dir,
                 load_model=args.load_model)
    else:
        nn = DRMM(config=conf)
        nn.test(test_point_file_path=args.test_file,
                test_size=args.test_size,
                output_file_path=args.output_score_file,
                load_model=True,
                emb_file_path=args.emb_file_path,
                checkpoint_dir=args.checkpoint_dir)