import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, Activation
from tensorflow.keras.layers import Input, Cropping1D, Flatten, Dropout, MaxPooling1D
from tensorflow.keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras import callbacks

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, MinMaxScaler

import os

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()

def hot_encode_seq(let):
    if let == 'A':
        return ([1, 0, 0, 0])
    elif let == 'T':
        return ([0, 1, 0, 0])
    elif let == 'C':
        return ([0, 0, 1, 0])
    elif let == 'G':
        return ([0, 0, 0, 1])
    elif let == 'N':
        return ([0, 0, 0, 0])

        
def rev_hot_encode_seq(let):
    y1 = [1, 0, 0, 0]
    y2 = [0, 1, 0, 0]
    y3 = [0, 0, 1, 0]
    y4 = [0, 0, 0, 1]
    y5 = [0, 0, 0, 0]
    results1 = all(map(lambda x, y: x == y, let, y1))
    results2 = all(map(lambda x, y: x == y, let, y2))
    results3 = all(map(lambda x, y: x == y, let, y3))
    results4 = all(map(lambda x, y: x == y, let, y4))
    results5 = all(map(lambda x, y: x == y, let, y5))
    if results1:
        return 'A'
    elif results2:
        return 'T'
    elif results3:
        return 'C'
    elif results4:
        return 'G'
    elif results5:
        return 'N'

def transform_label(labels):
    labels = labels.split(',')
    labels = [float(x) for x in labels]
    return labels


def transform_input(inputs_, labels_):
    inputs = []
    labels = []
    # hot-encode
    for i in range(len(inputs_)):
        printProgressBar(i, len(inputs_))
        # hot-encode seq
        inputs.append([np.array(hot_encode_seq(let)) for let in inputs_[i]])
        # CHANGE ONE-HOT ENCODING OF LABELS TO STRING->FLOATS
        try:
            labels.append(transform_label(labels_[i]))
        except ValueError:
            print(labels_[i])

    return inputs, labels

chr = ['chr1', 'chr3', 'chr5']

dir_inputs = './gdrive/MyDrive/dataset-new/inputs/positive/'
dir_labels = './gdrive/MyDrive/dataset-new/labels/positive/'

# <------------------- Positive G4 -------------------->
for chrN in chr:
    # <-------------- Minus strand --------------->
    filedir_input_minus = os.path.join(dir_inputs, chrN +'_inputs_minus.txt')
    filedir_labels_minus = os.path.join(dir_labels, chrN +'_minus_raw.txt')

    # import, process, concatenate with prev files
    if chrN == 'chr1':
        inputs_1 = np.loadtxt(filedir_input_minus, dtype='str', delimiter='\t')
        labels_1 = np.loadtxt(filedir_labels_minus, dtype='str', delimiter='\t')
        print('Shape of inputs, labels before transformation:', np.shape(inputs_1), np.shape(labels_1))

        inputs_1, labels_1 = transform_input(inputs_1, labels_1)
        inputs_1 = np.array(inputs_1)
        labels_1 = np.array(labels_1)
        labels_1 = np.array(np.amax(labels_1, axis=1))
        print('Shape of inputs, labels after transformation:', np.shape(inputs_1), np.shape(labels_1))

    else:
        inputs = np.loadtxt(filedir_input_minus, dtype='str', delimiter='\t')
        labels = np.loadtxt(filedir_labels_minus, dtype='str', delimiter='\t')
        print('Shape of inputs, labels before transformation:', np.shape(inputs), np.shape(labels))

        inputs, labels = transform_input(inputs, labels)
        inputs = np.array(inputs)
        labels = np.array(labels)
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # labels = scaler.fit_transform(labels)
        labels = np.array(np.amax(labels, axis=1))
        print('Shape of inputs, labels after transformation:', np.shape(inputs), np.shape(labels))

        inputs_1 = np.concatenate((inputs_1, inputs), axis=0)
        labels_1 = np.concatenate((labels_1, labels), axis=0)


for chrN in chr:
    # <-------------- Plus strand --------------->
    filedir_input_minus = os.path.join(dir_inputs, chrN +'_inputs_plus.txt')
    filedir_labels_minus = os.path.join(dir_labels, chrN +'_plus_raw.txt')

    # import, process, concatenate with prev files
    if chrN == 'chr1':
        inputs_2 = np.loadtxt(filedir_input_minus, dtype='str', delimiter='\t')
        labels_2 = np.loadtxt(filedir_labels_minus, dtype='str', delimiter='\t')
        print('Shape of inputs, labels before transformation:', np.shape(inputs_2), np.shape(labels_2))

        inputs_2, labels_2 = transform_input(inputs_2, labels_2)
        inputs_2 = np.array(inputs_2)
        labels_2 = np.array(labels_2)
        labels_2 = np.array(np.amax(labels_2, axis=1))
        print('Shape of inputs, labels after transformation:', np.shape(inputs_2), np.shape(labels_2))

    else:
        inputs = np.loadtxt(filedir_input_minus, dtype='str', delimiter='\t')
        labels = np.loadtxt(filedir_labels_minus, dtype='str', delimiter='\t')
        print('Shape of inputs, labels before transformation:', np.shape(inputs), np.shape(labels))

        inputs, labels = transform_input(inputs, labels)
        inputs = np.array(inputs)
        labels = np.array(labels)
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # labels = scaler.fit_transform(labels)
        labels = np.array(np.amax(labels, axis=1))
        print('Shape of inputs, labels after transformation:', np.shape(inputs), np.shape(labels))

        inputs_2 = np.concatenate((inputs_2, inputs), axis=0)
        labels_2 = np.concatenate((labels_2, labels), axis=0)

x_test_1 = np.concatenate((inputs_2, inputs_1), axis=0)
%reset_selective -f "^inputs_1$"
%reset_selective -f "^inputs_2$"

y_test_1 = np.concatenate((labels_2, labels_1), axis=0)
%reset_selective -f "^labels_1$"
%reset_selective -f "^labels_2$"



# <------------------- Negative G4 -------------------->
dir_inputs = './gdrive/MyDrive/dataset-new/inputs/random/'
dir_labels = './gdrive/MyDrive/dataset-new/labels/random/'

for chrN in chr:
    # <-------------- Plus strand --------------->
    filedir_input_minus = os.path.join(dir_inputs, chrN +'_inputs_plus.txt')
    filedir_labels_minus = os.path.join(dir_labels, chrN +'_plus_raw.txt')

    # import, process, concatenate with prev files
    if chrN == 'chr1':
        inputs_3 = np.loadtxt(filedir_input_minus, dtype='str', delimiter='\t')
        labels_3 = np.loadtxt(filedir_labels_minus, dtype='str', delimiter='\t')
        print('Shape of inputs, labels before transformation:', np.shape(inputs_3), np.shape(labels_3))

        inputs_3, labels_3 = transform_input(inputs_3, labels_3)
        inputs_3 = np.array(inputs_3)
        labels_3 = np.array(labels_3)
        labels_3 = np.array(np.amax(labels_3, axis=1))
        print('Shape of inputs, labels after transformation:', np.shape(inputs_3), np.shape(labels_3))

    else:
        inputs = np.loadtxt(filedir_input_minus, dtype='str', delimiter='\t')
        labels = np.loadtxt(filedir_labels_minus, dtype='str', delimiter='\t')
        print('Shape of inputs, labels before transformation:', np.shape(inputs), np.shape(labels))

        inputs, labels = transform_input(inputs, labels)
        inputs = np.array(inputs)
        labels = np.array(labels)
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # labels = scaler.fit_transform(labels)
        labels = np.array(np.amax(labels, axis=1))
        print('Shape of inputs, labels after transformation:', np.shape(inputs), np.shape(labels))

        inputs_3 = np.concatenate((inputs_3, inputs), axis=0)
        labels_3 = np.concatenate((labels_3, labels), axis=0)

for chrN in chr:
    # <-------------- Minus strand --------------->
    filedir_input_minus = os.path.join(dir_inputs, chrN +'_inputs_minus.txt')
    filedir_labels_minus = os.path.join(dir_labels, chrN +'_minus_raw.txt')

    # import, process, concatenate with prev files
    if chrN == 'chr1':
        inputs_4 = np.loadtxt(filedir_input_minus, dtype='str', delimiter='\t')
        labels_4 = np.loadtxt(filedir_labels_minus, dtype='str', delimiter='\t')
        print('Shape of inputs, labels before transformation:', np.shape(inputs_4), np.shape(labels_4))

        inputs_4, labels_4 = transform_input(inputs_4, labels_4)
        inputs_4 = np.array(inputs_4)
        labels_4 = np.array(labels_4)
        labels_4 = np.array(np.amax(labels_4, axis=1))
        print('Shape of inputs, labels after transformation:', np.shape(inputs_4), np.shape(labels_4))

    else:
        inputs = np.loadtxt(filedir_input_minus, dtype='str', delimiter='\t')
        labels = np.loadtxt(filedir_labels_minus, dtype='str', delimiter='\t')
        print('Shape of inputs, labels before transformation:', np.shape(inputs), np.shape(labels))

        inputs, labels = transform_input(inputs, labels)
        inputs = np.array(inputs)
        labels = np.array(labels)
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # labels = scaler.fit_transform(labels)
        labels = np.array(np.amax(labels, axis=1))
        print('Shape of inputs, labels after transformation:', np.shape(inputs), np.shape(labels))

        inputs_4 = np.concatenate((inputs_4, inputs), axis=0)
        labels_4 = np.concatenate((labels_4, labels), axis=0)

x_test_2 = np.concatenate((inputs_3, inputs_4), axis=0)
%reset_selective -f "^inputs_3$"
%reset_selective -f "^inputs_4$"

y_test_2 = np.concatenate((labels_3, labels_4), axis=0)
%reset_selective -f "^labels_4$"
%reset_selective -f "^labels_3$"

x_test = np.concatenate((x_test_1, x_test_2), axis=0)
%reset_selective -f "^inputs_1$"
%reset_selective -f "^inputs_2$"

y_test = np.concatenate((y_test_1, y_test_2), axis=0)
%reset_selective -f "^labels_1$"
%reset_selective -f "^labels_2$"

scaler = MinMaxScaler(feature_range=(0, 1))
y_test = y_test.reshape(-1, 1)
y_test = scaler.fit_transform(y_test)

print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

model_1label = keras.models.load_model('./gdrive/MyDrive/model/model-regression-onelabel-classparams.h5')
model_unfrozen = keras.models.load_model('./gdrive/MyDrive/model/model-finetune-unfrozen.h5')

model_1label.compile(loss=tf.keras.losses.MeanSquaredError(),
                     optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
                     metrics=tf.keras.metrics.MeanSquaredError())

model_unfrozen.compile(loss=tf.keras.losses.MeanSquaredError(),
                       optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                       metrics=tf.keras.metrics.MeanSquaredError())

print(model_1label.summary())
print(model_unfrozen.summary())

results_1label = model_1label.predict(x_test)
results_unfrozen = model_unfrozen.predict(x_test)

x = np.linspace(0, len(results_1label), num=len(results_1label)+1)
dif = [results_1label[i]-results_unfrozen[i] for i in range(len(results_1label))]
min_list = sorted(zip(x, dif), key=lambda t: t[1])
max_list = sorted(min_list, reverse=True)

min_idx, max_idx = [], []

for i in range(100):
  min_idx.append(min_list[i][0])

for i in range(len(min_list)-100, len(min_list)):
  max_idx.append(min_list[i][0])

for i in range(len(min_idx)):
  x = 0
  x = min_idx[i]
  x = int(x)
  min_idx[i] = x

for i in range(len(max_idx)):
  x = 0
  x = max_idx[i]
  x = int(x)
  max_idx[i] = x

min_score = []
for i in range(100):
  x = min_list[i][1].item()
  x = round(x, 2)
  min_score.append(x)

max_score = []
for i in range(100):
  x = max_list[i][1].item()
  x = round(x, 2)
  max_score.append(x)

with open('min_score.txt', 'w') as f:
    for item in min_score:
        f.write("%s\n" % item)

with open('max_score.txt', 'w') as f:
    for item in max_score:
        f.write("%s\n" % item)

min_seq, max_seq = [],[]

for j in range(len(min_idx)):
  n = min_idx[j]
  for i in range(150):
    min_seq.append(rev_hot_encode_seq(x_test[n][i]))

for j in range(len(max_idx)):
  n = max_idx[j]
  for i in range(150):
    max_seq.append(rev_hot_encode_seq(x_test[n][i]))

min_seq = np.reshape(min_seq, (100, 150))
max_seq = np.reshape(max_seq, (100, 150))

np.savetxt("min_seq.txt", min_seq, fmt="%s")
np.savetxt("max_seq.txt", max_seq, fmt="%s")
