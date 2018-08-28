from __future__ import division
from __future__ import print_function
# Basic python and data processing imports
import numpy as np

np.set_printoptions(suppress=True)  # Suppress scientific notation when printing small
# import h5py

import load_data_pairs as ld  # my scripts for loading data
import build_incept_model as bm  # Keras specification of SPEID model

# import matplotlib.pyplot as plt
from datetime import datetime
import util

# Keras imports
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.layers import Input, Convolution1D, MaxPooling1D, Merge, Dropout, Flatten, Dense, BatchNormalization, LSTM, \
    Activation, Bidirectional
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.models import Sequential
from keras.regularizers import l1, l2
from keras import backend as K

# cell_lines = ['GM12878', 'HeLa-S3', 'HUVEC', 'IMR90', 'K562', 'NHEK']
cell_lines = ['IMR90']
cell_lines_tests = ['GM12878', 'HeLa-S3', 'IMR90', 'K562']
# Model training parameters
num_epochs = 32
batch_size = 100
kernel_size = 96
training_frac = 0.9  # fraction of data to use for training

# Analysis the difference between the features from positive and negative groups based on the distance matrices
def dis_sim_matrix(seqA, seqB):
    num_samples, num_positions, num_features = seqA.shape
    num_samplesB, num_positionsB, num_featuresB = seqB.shape
    if ((num_samples != num_samplesB) | (num_features != num_featuresB)):
        print("The number of samples OR the number of feature maps is not matched!!")
    else:
        sim_matrix = np.zeros((num_positions, num_positionsB))

        for num in range(num_samples):
            sim_matrix_temp = np.zeros((num_positions, num_positionsB))
            for i in range(num_positions):
                for j in range(num_positionsB):
                    sim_matrix_temp[i, j] = np.linalg.norm(seqA[num, i, :], seqB[num, j, :])

            sim_matrix = sim_matrix + sim_matrix_temp

        sim_matrix = sim_matrix / num_samples

    return sim_matrix
t = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
opt = Adam(lr=1e-5)  # opt = RMSprop(lr = 1e-6)

data_path = '/home/panwei/zhuan143/all_cell_lines/'
out_path = data_path
for cell_line in cell_lines:
    print('Loading ' + cell_line + ' data from ' + data_path)
    X_enhancers = None
    X_promoters = None
    labels = None
    X_enhancers = np.load(out_path + cell_line + '_enhancers.npy')
    X_promoters = np.load(out_path + cell_line + '_promoters.npy')
    labels = np.load(out_path + cell_line + '_labels.npy')
    training_idx = np.random.randint(0, int(X_enhancers.shape[0]), size=27000)
    valid_idx = np.random.randint(0, int(X_enhancers.shape[0]), size=3000)
    X_enhancers_tr = X_enhancers[training_idx, :, :]
    X_promoters_tr = X_promoters[training_idx, :, :]
    labels_tr = labels[training_idx]
    X_enhancers_ts = X_enhancers[valid_idx, :, :]
    X_promoters_ts = X_promoters[valid_idx, :, :]
    labels_ts = labels[valid_idx]
    input_enhancer = Input((3000, 4))
    input_prompter = Input((2000, 4))
    #model = bm.build_inception_base(input_enhancer, input_prompter, seq_length_en=3000, seq_length_pro=2000)
    model = bm.build_inception_feature(input_enhancer, input_prompter, seq_length_en=3000, seq_length_pro=2000)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=["accuracy"])

    model.summary()


    # Define custom callback that prints/plots performance at end of each epoch
    class ConfusionMatrix(Callback):
        def on_train_begin(self, logs={}):
            self.epoch = 0
            self.precisions = []
            self.recalls = []
            self.f1_scores = []
            self.losses = []
            self.training_losses = []
            self.training_accs = []
            self.accs = []
            # plt.ion()

        def on_epoch_end(self, batch, logs={}):
            self.training_losses.append(logs.get('loss'))
            self.training_accs.append(logs.get('acc'))
            self.epoch += 1
            val_predict = model.predict_classes([X_enhancers, X_promoters], batch_size=batch_size, verbose=0)
            # util.print_live(self, labels, val_predict, logs)
            '''if self.epoch > 1: # need at least two time points to plot
                util.plot_live(self)'''


    # print '\nlabels.mean(): ' + str(labels.mean())
    print('Data sizes: ')
    print('[X_enhancers, X_promoters]: [' + str(np.shape(X_enhancers)) + ', ' + str(np.shape(X_promoters)) + ']')
    print('labels: ' + str(np.shape(labels)))

    # Instantiate callbacks
    confusionMatrix = ConfusionMatrix()
    # checkpoint_path = "/home/sss1/Desktop/projects/DeepInteractions/weights/test-delete-this-" + cell_line + "-basic-" + t + ".hdf5"
    # checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose = 1)

    print('Running fully trainable model for exactly ' + str(num_epochs) + ' epochs...')
    model.fit([X_enhancers_tr, X_promoters_tr],
              [labels_tr],
              validation_data=([X_enhancers_ts, X_promoters_ts], labels_ts),
              batch_size=batch_size,
              nb_epoch=num_epochs,
              shuffle=True
             # callbacks=[confusionMatrix]  # checkpointer]
              )

    print('Running predictions...')
    print('Running predictions...')
    for cell_line_test in cell_lines_tests:
        X_enhancers_test = np.load(out_path + cell_line_test + '_enhancers.npy')
        X_promoters_test = np.load(out_path + cell_line_test + '_promoters.npy')
        labels_test = np.load(out_path + cell_line_test + '_labels.npy')

        y_score = model.predict([X_enhancers_test, X_promoters_test], batch_size=50, verbose=1)
        np.save(('Basic_y_predict_Batch' + str(batch_size) + '_Kernel' + str(kernel_size) + cell_line + '_test' + cell_line_test), y_score)
    # np.save(('y_label'+cell_line), )

    inp = model.input
    outputs = [layer.output for layer in model.layers]
    functors = [K.function(inp+[K.learning_phase()], [out]) for out in outputs]

    # construct the positive and negative group separately
    test_shape = labels_test.shape[0]
    pos_num = 0
    for t in range(test_shape):
        if labels_test[t] == 1:
            pos_num = pos_num +1


    neg_num = int(test_shape-pos_num)

    ind_pos = np.zeros((pos_num))
    ind_neg = np.zeros((neg_num))
    pos_t = 0
    neg_t = 0
    for t in range(test_shape):
        if labels_test[t] == 1:
            ind_pos[pos_t] = int(t)
            pos_t = pos_t+1
        else:
            ind_neg[neg_t] = int(t)
            neg_t = neg_t+1

    X_enhancers_test_pos = X_enhancers_test[ind_pos, :, :]
    X_promoters_test_pos = X_promoters_test[ind_pos, :, :]
    labels_test_pos = labels_test[ind_pos]

    X_enhancers_test_neg = X_enhancers_test[ind_neg, :, :]
    X_promoters_test_neg = X_promoters_test[ind_neg, :, :]
    labels_test_neg = labels_test[ind_neg]

    # save the layers output from positive and negative inputs
    layers_out_pos = [func([[X_enhancers_test_pos, X_promoters_test_pos], labels_test_pos]) for func in functors]
    layers_out_neg = [func([[X_enhancers_test_neg, X_promoters_test_neg], labels_test_neg]) for func in functors]
    print('Layers_out_pos:')
    print(layers_out_pos)
    print('seqA:')
    seqA = model.layers['conv1d_1'].output
    print(seqA)
    print('seqB: ')
    seqB = model.promoter_branch.output
    print(seqB)




