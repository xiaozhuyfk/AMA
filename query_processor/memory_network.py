from __future__ import absolute_import

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class MemoryNetwork(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MemoryNetwork, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.W = np.identity(input_dim)

    def call(self, x, mask=None):
        return K.dot(x, self.W)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)
