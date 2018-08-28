from __future__ import print_function
import numpy as np

np.set_printoptions(suppress=True)
import h5py
import csv
#from sklearn.utils import shuffle
enhancers_length = 3000
promoters_length = 2000
num_bases = 4


cell_lines = ['GM12878', 'HeLa-S3', 'IMR90', 'K562', 'NHEK']

data_path = '/home/panwei/zhuan143/all_sequence_data.h5'
out_path = '/home/panwei/zhuan143/all_cell_lines/'

for cell_line in cell_lines:
    print ('Loading ' + cell_line + ' data from ' + data_path)
    X_enhancers = None
    X_promoters = None
    labels = None
    with h5py.File(data_path, 'r') as hf:
      X_enhancers = np.array(hf.get(cell_line + '_X_enhancers')).transpose((0, 2, 1))
      X_promoters = np.array(hf.get(cell_line + '_X_promoters')).transpose((0, 2, 1))
      labels = np.array(hf.get(cell_line + 'labels'))

    separate_path = (out_path + cell_line + '_enhancers.csv')
    np.savetxt(separate_path,X_enhancers)
    separate_path1 = (out_path + cell_line + '_promoters.csv')
    np.savetxt(separate_path1, X_promoters)
    separate_path2 = (out_path + cell_line + '_labels.csv')
    np.savetxt(separate_path2, labels)
    print('Writing filed finished !!')

    '''with h5py.File(separate_path,'w') as hf:
      hf.create_dataset('X_enhancers', data=X_enhancers)
      hf.create_dataset('X_promoters', data=X_promoters)
      hf.create_dataset('Labels', data=labels)'''




    '''file = open("all_sequence.txt","w")

      file.write(" Data sizes: ")
      file.write("[X_enhancers, X_promoters]: [" + str(np.shape(X_enhancers)) + ", " + str(np.shape(X_promoters)) + "]")
      file.write("labels: '"+ str(np.shape(labels)))'''
