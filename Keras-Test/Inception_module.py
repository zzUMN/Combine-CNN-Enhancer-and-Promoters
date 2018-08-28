from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from keras.layers import Input, Convolution1D, MaxPooling1D, Merge, Dropout, Flatten, Dense, BatchNormalization, LSTM, Activation, Bidirectional
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.models import Sequential
from keras.regularizers import l1, l2
filter_length = 40
class Inception_seq(Layer):

    def __init__(self,c1x1,c3x3,c5x5,p3x3 ):
        self.c1x1 = c1x1
        self.c3x3 = c3x3
        self.c5x5 = c5x5
        self.p3x3 = p3x3
        super(Inception_seq, self).__init__()

    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 4
        shape[1] = self.c1x1+self.c3x3+self.c5x5+self.p3x3

        return tuple(shape)

    def call(self, x, input_length, mask =None):
        conv1x1 = Convolution1D(input_dim = 4,
                                        input_length = input_length,
                                        nb_filter = self.c1x1,
                                        filter_length = filter_length,
                                        border_mode = "valid",
                                        subsample_length = 1,
                                        W_regularizer = l2(1e-5))
        conv3x3 = Convolution1D(input_dim=4,
                                input_length=input_length,
                                nb_filter=self.c3x3,
                                filter_length=filter_length,
                                border_mode="valid",
                                subsample_length=1,
                                W_regularizer=l2(1e-5))
        conv5x5 = Convolution1D(input_dim=4,
                                input_length=input_length,
                                nb_filter=self.c5x5,
                                filter_length=filter_length,
                                border_mode="valid",
                                subsample_length=1,
                                W_regularizer=l2(1e-5))

        pool3x3 =  MaxPooling1D(pool_length = filter_length/2, stride = filter_length/2)
