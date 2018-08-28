#from __future__ import division
from __future__ import print_function

import matplotlib as mp
mp.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, log_loss, roc_curve, auc, precision_recall_curve, average_precision_score
batch_size = 50
kernel_size = 96


def compute_AUPR(y, y_score):
  # print 'Computing Precision-Recall curve...'
  precision, recall, _ = precision_recall_curve(y, y_score)
  average_precision = average_precision_score(y, y_score)

def plot_PR_curve(y, y_score):
  # print 'Computing Precision-Recall curve...'
  precision, recall, _ = precision_recall_curve(y, y_score)
  return average_precision_score(y, y_score)

def plot_ROC_curve(y, y_score):
  # print 'Computing ROC curve...'
  fpr, tpr, thresholds = roc_curve(y, y_score)
  return auc(fpr, tpr), fpr, tpr

cell_lines_tests = ['GM12878', 'HeLa-S3', 'IMR90', 'K562']#, 'HUVEC', 'NHEK']
#cell_lines_tests = ['IMR90']
#cell_lines = ['GM12878', 'HeLa-S3','IMR90', 'K562', 'HUVEC', 'NHEK']
cell_lines = ['IMR90']
plt.figure()
for cell_line in cell_lines:

    for cell_line_test in cell_lines_tests:
        for r in range(1):

            y = np.array(np.load('/home/zzumn/Downloads/SimStruc-Test/' + cell_line_test + '_labels.npy'))
            index_test = np.random.randint(0, int(y.shape[0]), size=int(0.10 * y.shape[0]))
            y_score =np.array(np.load('/home/zzumn/Downloads/General-all-Test/Incept_feature_y_predict_Batch'+str(batch_size)+'_Kernel'+str(kernel_size)+cell_line+'_test'+cell_line_test+'.npy'))
            y_test = y[index_test]
            y_score_test = y_score[index_test]
            #y_test = y
            #y_score_test = y_score
            num_sample = y_score_test.shape[0]
            y_score_test = np.reshape(y_score_test, [num_sample, 1])
            print(cell_line+'_model test on '+cell_line_test)
            print('Area under PR Curve: ')
            print(plot_PR_curve(y, y_score))
            auc_p, fpr, tpr = plot_ROC_curve(y_test, y_score_test)
            print('Area under ROC Curve: ')
            print(auc_p)
            #plt.figure()
            lw = 2
            plt.plot(fpr, tpr, lw = lw, label=str(cell_line)+' added ROC curve (area= %0.2f)'% auc_p)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.05)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.title("Test on Inception Separated train Diagonal ROC Curve")
plt.show()


