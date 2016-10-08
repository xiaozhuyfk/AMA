from __future__ import absolute_import

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from math import exp

class MemoryNetwork(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MemoryNetwork, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.W = np.identity(input_dim)

    def call(self, x, mask=None):
        return x

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)


class FMNLayer(object):
    def __init__(self, triple):
        self.F = triple[0]
        self.q = triple[1]
        self.facts = triple[2]

    def predict(self):
        idx = np.argmax(self.Hnew)
        best_candidate = self.facts_new[idx]
        answer_sid = best_candidate[0]
        answer_r = best_candidate[2]

        A = []
        for (sid, s, r, oid, o) in self.facts_new:
            if sid == answer_sid and r == answer_r:
                A.append(o)
        return A

    def compute(self):
        H = self.score()
        Fnew, Hnew, facts_new = self.fact_prune(H)
        q = self.update(Fnew, Hnew)

        self.Fnew = Fnew
        self.Hnew = Hnew
        self.facts_new = facts_new

        return Fnew, q, facts_new

    def score(self):
        G = []
        H = []
        for f in self.F:
            (s, r, o) = f
            gscore = np.dot(self.q, (sum(s) + sum(r)))
            G.append(exp(gscore))
        Gsum = sum(G)
        for gscore in G:
            H.append(gscore / Gsum)
        return H

    def fact_prune(self, H):
        highest = max(H)
        Fnew = []
        Hnew = []
        facts_new = []
        for i in xrange(len(H)):
            h = H[i]
            if h > highest / 2:
                Fnew.append(self.F[i])
                Hnew.append(h)
                facts_new.append(self.facts[i])
        return Fnew, Hnew, facts_new

    def update(self, F, H):
        q = self.q
        for i in xrange(len(F)):
            (s, r, o) = F[i]
            h = H[i]
            q += h * (s + r)
        return q

