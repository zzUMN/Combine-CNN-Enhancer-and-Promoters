from __future__ import division
from __future__ import print_function
# Basic python and data processing imports
import numpy as np
np.set_printoptions(suppress=True) # Suppress scientific notation when printing small
#import h5py

import load_data_pairs as ld # my scripts for loading data
import build_sim_model as bm # Keras specification of SPEID model
#import build_module_model as bm
# import matplotlib.pyplot as plt
from datetime import datetime
import util

# Keras imports
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback

cell_lines = ['GM12878', 'HeLa-S3', 'HUVEC', 'IMR90', 'K562', 'NHEK']
#cell_lines = ['GM12878','HeLa-S3','K562', 'IMR90']
cell_lines_spec = ['GM12878']
cell_lines_tests = ['GM12878', 'HeLa-S3', 'K562', 'IMR90', 'NHEK', 'HUVEC']
# Model training parameters
num_epochs = 22
num_epochs_pre = 10
batch_size = 100
kernel_size  = 300
training_frac = 0.9 # fraction of data to use for training

t = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
opt = Adam(lr = 1e-5) # opt = RMSprop(lr = 1e-6)

data_path = '/home/panwei/zhuan143/all_cell_lines/'
out_path = data_path
for ro in range(1):# several rounds to test the stablity of the model OR accumulate the model's performance
    for cell_line in cell_lines:

      np.random.seed(ro+10)
      print('Loading ' + cell_line + ' data from ' + data_path)
      X_enhancers = None
      X_promoters = None
      labels = None
      X_enhancers = np.load(out_path + cell_line + '_enhancers.npy')
      X_promoters = np.load(out_path + cell_line + '_promoters.npy')
      labels = np.load(out_path + cell_line + '_labels.npy')
      training_idx = np.random.randint(0,int(X_enhancers.shape[0]), size=int(0.81*X_enhancers.shape[0]))
      valid_idx = np.random.randint(0, int(X_enhancers.shape[0]), size=int(0.09*X_enhancers.shape[0]))
      X_enhancers_tr = X_enhancers[training_idx,:,:]
      X_promoters_tr = X_promoters[training_idx,:,:]
      labels_tr = labels[training_idx]
      X_enhancers_ts = X_enhancers[valid_idx,:,:]
      X_promoters_ts = X_promoters[valid_idx,:,:]
      labels_ts = labels[valid_idx]
      model = bm.build_model(use_JASPAR = False)

      model.compile(loss = 'binary_crossentropy',
                    optimizer = opt,
                    metrics = ["accuracy"])

      model.summary()


      # Define custom callback that prints/plots performance at end of each epoch
      class ConfusionMatrix(Callback):
          def on_train_begin(self, logs = {}):
              self.epoch = 0
              self.precisions = []
              self.recalls = []
              self.f1_scores = []
              self.losses = []
              self.training_losses = []
              self.training_accs = []
              self.accs = []
              #plt.ion()

          def on_epoch_end(self, batch, logs = {}):
              self.training_losses.append(logs.get('loss'))
              self.training_accs.append(logs.get('acc'))
              self.epoch += 1
              val_predict = model.predict_classes([X_enhancers, X_promoters], batch_size = batch_size, verbose = 0)
              #util.print_live(self, labels, val_predict, logs)
              '''if self.epoch > 1: # need at least two time points to plot
                  util.plot_live(self)'''

      # print '\nlabels.mean(): ' + str(labels.mean())
      print('Data sizes: ')
      print('[X_enhancers, X_promoters]: [' + str(np.shape(X_enhancers)) + ', ' + str(np.shape(X_promoters)) + ']')
      print('labels: ' + str(np.shape(labels)))

      # Instantiate callbacks
      confusionMatrix = ConfusionMatrix()
      #checkpoint_path = "/home/sss1/Desktop/projects/DeepInteractions/weights/test-delete-this-" + cell_line + "-basic-" + t + ".hdf5"
      #checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose = 1)

      print('Running fully trainable model for exactly ' + str(num_epochs_pre) + ' epochs...')
      model.fit([X_enhancers_tr, X_promoters_tr],
                  [labels_tr],
                  validation_data = ([X_enhancers_ts, X_promoters_ts], labels_ts),
                  batch_size = batch_size,
                  nb_epoch = num_epochs_pre,
                  shuffle = True,
                  callbacks=[confusionMatrix] #checkpointer]
                  )

   #Retrain the model with the data from specific cell line
    print('Running fully trainable model over specified cell line for exactly ' + str(num_epochs) + ' epochs...')
    X_enhancers_spec = np.load(out_path + cell_lines_spec + '_enhancers.npy')
    X_promoters_spec = np.load(out_path + cell_lines_spec + '_promoters.npy')
    labels_spec = np.load(out_path + cell_lines_spec + '_labels.npy')
    training_idx = np.random.randint(0, int(X_enhancers_spec.shape[0]), size=int(0.81 * X_enhancers_spec.shape[0]))
    valid_idx = np.random.randint(0, int(X_enhancers_spec.shape[0]), size=int(0.09 * X_enhancers_spec.shape[0]))
    X_enhancers_tr_spec = X_enhancers_spec[training_idx, :, :]
    X_promoters_tr_spec = X_promoters_spec[training_idx, :, :]
    labels_tr_spec = labels[training_idx]
    X_enhancers_ts_spec = X_enhancers_spec[valid_idx, :, :]
    X_promoters_ts_spec = X_promoters_spec[valid_idx, :, :]
    labels_ts_spec = labels_spec[valid_idx]

    print('Building frozen model ......')
    model = bm.build_frozen_model()
    model.compile(loss= 'bianry_crossentropy', optimizer=opt, metrics=["accuracy"])
    confusionMatrixFrozen = ConfusionMatrix()

    model.fit([X_enhancers_tr_spec, X_promoters_tr_spec],
              [labels_tr_spec],
              validation_data=([X_enhancers_ts_spec, X_promoters_ts_spec], labels_ts_spec),
              batch_size=batch_size,
              nb_epoch=num_epochs,
              shuffle=True,
              callbacks=[confusionMatrixFrozen]  # checkpointer]
              )


    print('Running predictions...')
    for cell_line_test in cell_lines_tests:# monitor the process of training on all the cell lines to make a general model and test on all the cell lines
        X_enhancers_test = np.load(out_path + cell_line_test + '_enhancers.npy')
        X_promoters_test = np.load(out_path + cell_line_test + '_promoters.npy')
        labels_test = np.load(out_path + cell_line_test + '_labels.npy')

        y_score = model.predict([X_enhancers_test, X_promoters_test], batch_size = 50, verbose = 1)
        np.save(('Frozen_general_y_predict_Batch'+str(batch_size)+'_Kernel'+str(kernel_size)+cell_line+'_test'+cell_line_test+str(ro)), y_score)
