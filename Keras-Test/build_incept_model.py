import numpy as np

import warnings

from keras.layers.convolutional import MaxPooling1D, Convolution1D, AveragePooling1D
from keras.layers import Input, Dropout, Dense, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras import regularizers
from keras import initializers
from keras.models import Model

from keras import backend as K

from keras.utils.layer_utils import convert_all_kernels_in_model
#from keras.utils.data.utils import get_file
dense_layer_size = 1000
def prepocess_input(x):
    x = np.divide(x,255.0)
    x = np.subtract(x, 0.5)
    x = np.multiply(x, 2.0)

    return x

def conv1d_bn(x, nb_filter, filter_length, seq_length, padding= 'same', strides = 1, use_bias=False):
    x = Convolution1D(input_dim = 4,
                            input_length = seq_length,
                            nb_filter = nb_filter,
                            filter_length = filter_length,
                            border_mode = padding,
                            subsample_length = 1,
                            W_regularizer = regularizers.l2(1e-5))(x)
    x = BatchNormalization(momentum=0.997, scale=False)(x)
    x = Activation('relu')(x)

    return x

def block_inception_a(input, nb_filter, filter_length, seq_length):
    branch_0 = conv1d_bn(input, nb_filter, filter_length, seq_length)

    branch_1 = conv1d_bn(input, 44, 1, seq_length)
    branch_1 = conv1d_bn(branch_1, 64, 3, seq_length)

    branch_2 = conv1d_bn(input, 44, 1, seq_length)
    branch_2 = conv1d_bn(branch_2, 64, 3, seq_length)
    branch_2 = conv1d_bn(branch_2, 64, 3, seq_length)

    branch_3 = AveragePooling1D(3,strides=1,padding='same')(input)
    branch_3 = conv1d_bn(branch_3, 64, 1, seq_length)

    x = concatenate([branch_0, branch_1, branch_2, branch_3])

    return x

def block_reduction_a(input, seq_length):
    branch_0 = conv1d_bn(input, 256, 3, seq_length=seq_length, strides=2, padding='valid')

    branch_1 = conv1d_bn(input, 128, 1, seq_length=seq_length)
    branch_1 = conv1d_bn(branch_1, 150, 3, seq_length=seq_length)
    branch_1 = conv1d_bn(branch_1,256, 3, seq_length=seq_length, strides=2, padding='valid')

    branch_2 = MaxPooling1D(pool_size=3, strides=2,padding='valid')(input)

    x = concatenate([branch_0, branch_1, branch_2],axis=1)

    return x

def block_inception_b(input, seq_length):
    branch_0 = conv1d_bn(input, 48, 20, seq_length)

    branch_1 = conv1d_bn(input, 32, 20, seq_length)
    branch_1 = conv1d_bn(branch_1, 48, 60, seq_length)

    branch_2 = conv1d_bn(input, 32, 20, seq_length)
    branch_2 = conv1d_bn(branch_2, 48, 60, seq_length)
    branch_2 = conv1d_bn(branch_2, 48, 60, seq_length)

    branch_3 = AveragePooling1D(60,strides=1,padding='same')(input)
    branch_3 = conv1d_bn(branch_3, 48,20,seq_length)

    x = concatenate([branch_0, branch_1, branch_2, branch_3])

    return x

def block_reduction_b(input, seq_length):
    branch_0 = conv1d_bn(input, 192, 60, seq_length=seq_length, strides=2, padding='valid')

    branch_1 = conv1d_bn(input, 96, 20, seq_length=seq_length)
    branch_1 = conv1d_bn(branch_1, 112, 60, seq_length=seq_length)
    branch_1 = conv1d_bn(branch_1, 192, 60, seq_length=seq_length, strides=2, padding='valid')

    branch_2 = MaxPooling1D(pool_size=60, strides=2,padding='valid')(input)

    x = concatenate([branch_0, branch_1, branch_2],axis=1)

    return x

def build_inception_base(inputs_en, inputs_pro, seq_length_en, seq_length_pro):
    enhancer_branch = conv1d_bn(inputs_en, 1024, 40, seq_length=seq_length_en)
    enhancer_branch = MaxPooling1D(pool_size=20, strides=20)(enhancer_branch)

    promoter_branch = conv1d_bn(inputs_pro, 1024, 40, seq_length=seq_length_pro)
    promoter_branch = MaxPooling1D(pool_size=20, strides=20)(promoter_branch)

    merge_feature = concatenate([enhancer_branch, promoter_branch],axis=1)

    #net = Flatten()(merge_feature)
    net = block_inception_a(merge_feature, 96, 1, seq_length=int((seq_length_en+seq_length_pro)/20))

    net = block_reduction_a(net, seq_length=int((seq_length_en+seq_length_pro)/20))

    net = Flatten()(net)
    net = Dense(output_dim=dense_layer_size,
                    init="glorot_uniform",
                    W_regularizer=regularizers.l2(1e-6))(net)
    net = BatchNormalization(momentum=0.997, scale=False)(net)
    net = Activation('relu')(net)

    net = Dense(output_dim=1)(net)
    net = BatchNormalization(momentum=0.997, scale=False)(net)
    net = Activation('sigmoid')(net)
    model = Model([inputs_en, inputs_pro], net)
    
    return model

def build_inception_feature(inputs_en, inputs_pro, seq_length_en, seq_length_pro):
    enhancer_branch = block_inception_b(inputs_en, 3000)
    enhancer_branch = block_reduction_b(enhancer_branch, 3000)

    promoter_branch = block_inception_b(inputs_pro, 2000)
    promoter_branch = block_reduction_b(promoter_branch, 2000)

    merge_feature = concatenate([enhancer_branch, promoter_branch],axis=1)
    net = Flatten()(merge_feature)
    net = Dense(output_dim=dense_layer_size,
                init="glorot_uniform",
                W_regularizer=regularizers.l2(1e-6))(net)
    net = BatchNormalization(momentum=0.997, scale=False)(net)
    net = Activation('relu')(net)

    net = Dense(output_dim=1)(net)
    net = BatchNormalization(momentum=0.997, scale=False)(net)
    net = Activation('sigmoid')(net)
    model = Model([inputs_en, inputs_pro], net)

    return model

def build_shared_projection(inputs_en, inputs_pro, seq_length_en, seq_length_pro):
    enhancer_branch = conv1d_bn(inputs_en, 1024, 40, seq_length=seq_length_en)
    enhancer_branch = MaxPooling1D(pool_size=20, strides=20)(enhancer_branch)

    promoter_branch = conv1d_bn(inputs_pro, 1024, 40, seq_length=seq_length_pro)
    promoter_branch = MaxPooling1D(pool_size=20, strides=20)(promoter_branch)

    merge_feature = concatenate([enhancer_branch, promoter_branch], axis=1)

    # net = Flatten()(merge_feature)

    net = Flatten()(merge_feature)
    net = Dense(output_dim=dense_layer_size,
                init="glorot_uniform",
                W_regularizer=regularizers.l2(1e-6))(net)
    net = BatchNormalization(momentum=0.997, scale=False)(net)
    net = Activation('relu')(net)

    net = Dense(output_dim=1)(net)
    net = BatchNormalization(momentum=0.997, scale=False)(net)
    net = Activation('sigmoid')(net)

    # shared projection task 1
    net1 = Flatten()(enhancer_branch)
    net1 =  Dense(output_dim=dense_layer_size,
                init="glorot_uniform",
                W_regularizer=regularizers.l2(1e-6))(net1)
    net1 = BatchNormalization(momentum=0.997, scale=False)(net1)
    net1 = Activation('relu')(net1)

    net1 = Dense(output_dim=seq_length_pro)(net1)
    net1 = BatchNormalization(momentum=0.997, scale=False)(net1)
    net1 = Activation('sigmoid')(net1)
    # shared projection task 2
    net2 = Flatten()(promoter_branch)
    net2 = Dense(output_dim=dense_layer_size,
                init="glorot_uniform",
                W_regularizer=regularizers.l2(1e-6))(net2)
    net2 = BatchNormalization(momentum=0.997, scale=False)(net2)
    net2 = Activation('relu')(net2)

    net2 = Dense(output_dim=seq_length_pro)(net2)
    net2 = BatchNormalization(momentum=0.997, scale=False)(net2)
    net2 = Activation('sigmoid')(net2)



    model = Model([inputs_en, inputs_pro], [net, net1, net2])

    return model

