import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, Activation, Input, Cropping1D, Embedding, LSTM
from tensorflow.keras.models import Model
from tensorflow.python.ops import math_ops

import sklearn
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import normalize, MinMaxScaler

import keras
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler

import os
import time

def lr_schedule(epoch):
    lr = 0.001
    if epoch == 3:
        lr *= 0.5
    elif epoch == 4:
        lr *= 0.5 ** 2
    elif epoch == 5:
        lr *= 0.5 ** 3
    elif epoch == 7:
        lr *= 0.5 ** 4
    print('Learning rate: ', lr)
    return lr


def RB_block(inputs,
             num_filters=32,
             kernel_size=11,
             strides=1,
             activation='relu',
             dilation_rate=1):
    """1D Convolution-Batch Normalization-Activation stack builder
    """
    conv = Conv1D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  dilation_rate=dilation_rate)
    x = inputs
    for layer in range(2):
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = conv(x)
    return x


def model_CNN_full(input_shape, rb, dil, kernel_size):
    """Model builder
    """
    inputs = Input(shape=input_shape)

    # initiate
    x = Conv1D(32, kernel_size=1, strides=1, padding='same', dilation_rate=1)(inputs)
    # another Conv on x before splitting
    y = Conv1D(32, kernel_size=1, strides=1, padding='same', dilation_rate=1)(x)

    d = [1, dil]  # dilation
    for i in range(2):
        for stack in range(rb):
            x = RB_block(x, num_filters=32, kernel_size=kernel_size, strides=1, activation='relu', dilation_rate=d[i])
        if i == 0:
            y = keras.layers.add([Conv1D(32, kernel_size=1, strides=1, padding='same', dilation_rate=1)(x), y])

    x = Conv1D(32, kernel_size=1, strides=1, padding='same', dilation_rate=1)(x)
    # adding up with what was shortcut from the prev layers
    x = keras.layers.add([x, y])
    x = Conv1D(1, kernel_size=1, strides=1, padding='same', dilation_rate=1)(x)
    x = Dense(1, activation='sigmoid')(x)
    outputs = Cropping1D(cropping=(25, 25))(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


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

chromosomes_1 = ['chr2', 'chr4', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10']
chromosomes_2 = ['chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17']
chromosomes_3 = ['chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY']        

# <----------------- Positive Sequence ------------------->
dir_inputs = './gdrive/MyDrive/dataset-new/inputs/positive/'
dir_labels = './gdrive/MyDrive/dataset-new/labels/positive/'

for chrN in chromosomes_3:
    # <-------------- Minus strand --------------->
    filedir_input_minus = os.path.join(dir_inputs, chrN +'_inputs_minus.txt')
    filedir_labels_minus = os.path.join(dir_labels, chrN +'_minus_raw.txt')

    # import, process, concatenate with prev files
    if chrN == 'chr18':
        inputs_1 = np.loadtxt(filedir_input_minus, dtype='str', delimiter='\t')
        labels_1 = np.loadtxt(filedir_labels_minus, dtype='str', delimiter='\t')
        print('Shape of inputs, labels before transformation:', np.shape(inputs_1), np.shape(labels_1))

        inputs_1, labels_1 = transform_input(inputs_1, labels_1)
        inputs_1 = np.array(inputs_1)
        labels_1 = np.array(labels_1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        labels_1 = scaler.fit_transform(labels_1)
        print('Shape of inputs, labels after transformation:', np.shape(inputs_1), np.shape(labels_1))

    else:
        inputs = np.loadtxt(filedir_input_minus, dtype='str', delimiter='\t')
        labels = np.loadtxt(filedir_labels_minus, dtype='str', delimiter='\t')
        print('Shape of inputs, labels before transformation:', np.shape(inputs), np.shape(labels))

        inputs, labels = transform_input(inputs, labels)
        inputs = np.array(inputs)
        labels = np.array(labels)
        scaler = MinMaxScaler(feature_range=(0, 1))
        labels = scaler.fit_transform(labels)
        print('Shape of inputs, labels after transformation:', np.shape(inputs), np.shape(labels))

        inputs_1 = np.concatenate((inputs_1, inputs), axis=0)
        labels_1 = np.concatenate((labels_1, labels), axis=0)

for chrN in chromosomes_3:
    # <-------------- Plus strand --------------->
    filedir_input_plus = os.path.join(dir_inputs, chrN + '_inputs_plus.txt')
    filedir_labels_plus = os.path.join(dir_labels, chrN + '_plus_raw.txt')

    # import, process, concatenate with prev files
    if chrN == 'chr18':
        inputs_2 = np.loadtxt(filedir_input_plus, dtype='str', delimiter='\t')
        labels_2 = np.loadtxt(filedir_labels_plus, dtype='str', delimiter='\t')
        print('Shape of inputs, labels before transformation:', np.shape(inputs_2), np.shape(labels_2))

        inputs_2, labels_2 = transform_input(inputs_2, labels_2)
        inputs_2 = np.array(inputs_2)
        labels_2 = np.array(labels_2)
        scaler = MinMaxScaler(feature_range=(0, 1))
        labels_2 = scaler.fit_transform(labels_2)
        print('Shape of inputs, labels after transformation:', np.shape(inputs_2), np.shape(labels_2))

    else:
        inputs = np.loadtxt(filedir_input_plus, dtype='str', delimiter='\t')
        labels = np.loadtxt(filedir_labels_plus, dtype='str', delimiter='\t')
        print('Shape of inputs, labels before transformation:', np.shape(inputs), np.shape(labels))

        inputs, labels = transform_input(inputs, labels)
        inputs = np.array(inputs)
        labels = np.array(labels)
        scaler = MinMaxScaler(feature_range=(0, 1))
        labels = scaler.fit_transform(labels)
        print('Shape of inputs, labels after transformation:', np.shape(inputs), np.shape(labels))

        inputs_2 = np.concatenate((inputs_2, inputs), axis=0)
        labels_2 = np.concatenate((labels_2, labels), axis=0)

x_train_1 = np.concatenate((inputs_2, inputs_1), axis=0)
%reset_selective -f "^inputs_1$"
%reset_selective -f "^inputs_2$"

y_train_1 = np.concatenate((labels_2, labels_1), axis=0)
%reset_selective -f "^labels_1$"
%reset_selective -f "^labels_2$"

# <----------------- Control Sequence ------------------->
dir_inputs = './gdrive/MyDrive/dataset-new/inputs/random/'
dir_labels = './gdrive/MyDrive/dataset-new/labels/random/'

for chrN in chromosomes_3:
    # <-------------- Minus strand --------------->
    filedir_input_minus = os.path.join(dir_inputs, chrN +'_inputs_minus.txt')
    filedir_labels_minus = os.path.join(dir_labels, chrN +'_minus_raw.txt')

    # import, process, concatenate with prev files
    if chrN == 'chr18':
        inputs_3 = np.loadtxt(filedir_input_minus, dtype='str', delimiter='\t')
        labels_3 = np.loadtxt(filedir_labels_minus, dtype='str', delimiter='\t')
        print('Shape of inputs, labels before transformation:', np.shape(inputs_3), np.shape(labels_3))

        inputs_3, labels_3 = transform_input(inputs_3, labels_3)
        inputs_3 = np.array(inputs_3)
        labels_3 = np.array(labels_3)
        scaler = MinMaxScaler(feature_range=(0, 1))
        labels_3 = scaler.fit_transform(labels_3)
        print('Shape of inputs, labels after transformation:', np.shape(inputs_3), np.shape(labels_3))

    else:
        inputs = np.loadtxt(filedir_input_minus, dtype='str', delimiter='\t')
        labels = np.loadtxt(filedir_labels_minus, dtype='str', delimiter='\t')
        print('Shape of inputs, labels before transformation:', np.shape(inputs), np.shape(labels))

        inputs, labels = transform_input(inputs, labels)
        inputs = np.array(inputs)
        labels = np.array(labels)
        scaler = MinMaxScaler(feature_range=(0, 1))
        labels = scaler.fit_transform(labels)
        print('Shape of inputs, labels after transformation:', np.shape(inputs), np.shape(labels))

        inputs_3 = np.concatenate((inputs_3, inputs), axis=0)
        labels_3 = np.concatenate((labels_3, labels), axis=0)

for chrN in chromosomes_3:
    # <-------------- Plus strand --------------->
    filedir_input_plus = os.path.join(dir_inputs, chrN + '_inputs_plus.txt')
    filedir_labels_plus = os.path.join(dir_labels, chrN + '_plus_raw.txt')

    # import, process, concatenate with prev files
    if chrN == 'chr18':
        inputs_4 = np.loadtxt(filedir_input_plus, dtype='str', delimiter='\t')
        labels_4 = np.loadtxt(filedir_labels_plus, dtype='str', delimiter='\t')
        print('Shape of inputs, labels before transformation:', np.shape(inputs_4), np.shape(labels_4))

        inputs_4, labels_4 = transform_input(inputs_4, labels_4)
        inputs_4 = np.array(inputs_4)
        labels_4 = np.array(labels_4)
        scaler = MinMaxScaler(feature_range=(0, 1))
        labels_4 = scaler.fit_transform(labels_4)
        print('Shape of inputs, labels after transformation:', np.shape(inputs_4), np.shape(labels_4))

    else:
        inputs = np.loadtxt(filedir_input_plus, dtype='str', delimiter='\t')
        labels = np.loadtxt(filedir_labels_plus, dtype='str', delimiter='\t')
        print('Shape of inputs, labels before transformation:', np.shape(inputs), np.shape(labels))

        inputs, labels = transform_input(inputs, labels)
        inputs = np.array(inputs)
        labels = np.array(labels)
        scaler = MinMaxScaler(feature_range=(0, 1))
        labels = scaler.fit_transform(labels)
        print('Shape of inputs, labels after transformation:', np.shape(inputs), np.shape(labels))

        inputs_4 = np.concatenate((inputs_4, inputs), axis=0)
        labels_4 = np.concatenate((labels_4, labels), axis=0)

x_train_2 = np.concatenate((inputs_3, inputs_4), axis=0)
%reset_selective -f "^inputs_3$"
%reset_selective -f "^inputs_4$"

y_train_2 = np.concatenate((labels_3, labels_4), axis=0)
%reset_selective -f "^labels_4$"
%reset_selective -f "^labels_3$"

x_train = np.concatenate((x_train_1, x_train_2), axis=0)
%reset_selective -f "^x_train_1$"
%reset_selective -f "^x_train_2$"

y_train = np.concatenate((y_train_1, y_train_2), axis=0)
%reset_selective -f "^y_train_1$"
%reset_selective -f "^y_train_2$"

input_shape = x_train.shape[1:]
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print('y_train shape:', y_train.shape)

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "./gdrive/MyDrive/model/training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


# Create a callback that saves the model's weights every epoch
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model = model_CNN_full(input_shape=input_shape, rb=3, dil=8, kernel_size=5)

model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=Adam(learning_rate=0.001),
              metrics=tf.keras.metrics.MeanSquaredError())

print(model.summary())

# Save the weights using the `checkpoint_path` format
# model.load_weights('./gdrive/MyDrive/model/g4nn-cnn-2.h5')
# model.save_weights(checkpoint_path.format(epoch=0))

history = model.fit(x_train, y_train, epochs=5, batch_size=40, shuffle=True, callbacks=[cp_callback], validation_split=0.2)

model.save('./gdrive/MyDrive/model/model-regression.h5')

%reset_selective -f "^x_train$"
%reset_selective -f "^y_train$"

chromosomes = ['chr1', 'chr3', 'chr5']

# <----------------- Positive Sequence ------------------->
dir_inputs = './gdrive/MyDrive/dataset-new/inputs/positive/'
dir_labels = './gdrive/MyDrive/dataset-new/labels/positive/'

for chrN in chromosomes:
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
        scaler = MinMaxScaler(feature_range=(0, 1))
        labels_1 = scaler.fit_transform(labels_1)
        print('Shape of inputs, labels after transformation:', np.shape(inputs_1), np.shape(labels_1))

    else:
        inputs = np.loadtxt(filedir_input_minus, dtype='str', delimiter='\t')
        labels = np.loadtxt(filedir_labels_minus, dtype='str', delimiter='\t')
        print('Shape of inputs, labels before transformation:', np.shape(inputs), np.shape(labels))

        inputs, labels = transform_input(inputs, labels)
        inputs = np.array(inputs)
        labels = np.array(labels)
        scaler = MinMaxScaler(feature_range=(0, 1))
        labels = scaler.fit_transform(labels)
        print('Shape of inputs, labels after transformation:', np.shape(inputs), np.shape(labels))

        inputs_1 = np.concatenate((inputs_1, inputs), axis=0)
        labels_1 = np.concatenate((labels_1, labels), axis=0)

for chrN in chromosomes:
    # <-------------- Plus strand --------------->
    filedir_input_plus = os.path.join(dir_inputs, chrN + '_inputs_plus.txt')
    filedir_labels_plus = os.path.join(dir_labels, chrN + '_plus_raw.txt')

    # import, process, concatenate with prev files
    if chrN == 'chr1':
        inputs_2 = np.loadtxt(filedir_input_plus, dtype='str', delimiter='\t')
        labels_2 = np.loadtxt(filedir_labels_plus, dtype='str', delimiter='\t')
        print('Shape of inputs, labels before transformation:', np.shape(inputs_2), np.shape(labels_2))

        inputs_2, labels_2 = transform_input(inputs_2, labels_2)
        inputs_2 = np.array(inputs_2)
        labels_2 = np.array(labels_2)
        scaler = MinMaxScaler(feature_range=(0, 1))
        labels_2 = scaler.fit_transform(labels_2)
        print('Shape of inputs, labels after transformation:', np.shape(inputs_2), np.shape(labels_2))

    else:
        inputs = np.loadtxt(filedir_input_plus, dtype='str', delimiter='\t')
        labels = np.loadtxt(filedir_labels_plus, dtype='str', delimiter='\t')
        print('Shape of inputs, labels before transformation:', np.shape(inputs), np.shape(labels))

        inputs, labels = transform_input(inputs, labels)
        inputs = np.array(inputs)
        labels = np.array(labels)
        scaler = MinMaxScaler(feature_range=(0, 1))
        labels = scaler.fit_transform(labels)
        print('Shape of inputs, labels after transformation:', np.shape(inputs), np.shape(labels))

        inputs_2 = np.concatenate((inputs_2, inputs), axis=0)
        labels_2 = np.concatenate((labels_2, labels), axis=0)

x_test_1 = np.concatenate((inputs_2, inputs_1), axis=0)
%reset_selective -f "^inputs_1$"
%reset_selective -f "^inputs_2$"

y_test_1 = np.concatenate((labels_2, labels_1), axis=0)
%reset_selective -f "^labels_1$"
%reset_selective -f "^labels_2$"

# <----------------- Control Sequence ------------------->
dir_inputs = './gdrive/MyDrive/dataset-new/inputs/random/'
dir_labels = './gdrive/MyDrive/dataset-new/labels/random/'

for chrN in chromosomes:
    # <-------------- Minus strand --------------->
    filedir_input_minus = os.path.join(dir_inputs, chrN +'_inputs_minus.txt')
    filedir_labels_minus = os.path.join(dir_labels, chrN +'_minus_raw.txt')

    # import, process, concatenate with prev files
    if chrN == 'chr1':
        inputs_3 = np.loadtxt(filedir_input_minus, dtype='str', delimiter='\t')
        labels_3 = np.loadtxt(filedir_labels_minus, dtype='str', delimiter='\t')
        print('Shape of inputs, labels before transformation:', np.shape(inputs_3), np.shape(labels_3))

        inputs_3, labels_3 = transform_input(inputs_3, labels_3)
        inputs_3 = np.array(inputs_3)
        labels_3 = np.array(labels_3)
        scaler = MinMaxScaler(feature_range=(0, 1))
        labels_3 = scaler.fit_transform(labels_3)
        print('Shape of inputs, labels after transformation:', np.shape(inputs_3), np.shape(labels_3))

    else:
        inputs = np.loadtxt(filedir_input_minus, dtype='str', delimiter='\t')
        labels = np.loadtxt(filedir_labels_minus, dtype='str', delimiter='\t')
        print('Shape of inputs, labels before transformation:', np.shape(inputs), np.shape(labels))

        inputs, labels = transform_input(inputs, labels)
        inputs = np.array(inputs)
        labels = np.array(labels)
        scaler = MinMaxScaler(feature_range=(0, 1))
        labels = scaler.fit_transform(labels)
        print('Shape of inputs, labels after transformation:', np.shape(inputs), np.shape(labels))

        inputs_3 = np.concatenate((inputs_3, inputs), axis=0)
        labels_3 = np.concatenate((labels_3, labels), axis=0)

for chrN in chromosomes:
    # <-------------- Plus strand --------------->
    filedir_input_plus = os.path.join(dir_inputs, chrN + '_inputs_plus.txt')
    filedir_labels_plus = os.path.join(dir_labels, chrN + '_plus_raw.txt')

    # import, process, concatenate with prev files
    if chrN == 'chr1':
        inputs_4 = np.loadtxt(filedir_input_plus, dtype='str', delimiter='\t')
        labels_4 = np.loadtxt(filedir_labels_plus, dtype='str', delimiter='\t')
        print('Shape of inputs, labels before transformation:', np.shape(inputs_4), np.shape(labels_4))

        inputs_4, labels_4 = transform_input(inputs_4, labels_4)
        inputs_4 = np.array(inputs_4)
        labels_4 = np.array(labels_4)
        scaler = MinMaxScaler(feature_range=(0, 1))
        labels_4 = scaler.fit_transform(labels_4)
        print('Shape of inputs, labels after transformation:', np.shape(inputs_4), np.shape(labels_4))

    else:
        inputs = np.loadtxt(filedir_input_plus, dtype='str', delimiter='\t')
        labels = np.loadtxt(filedir_labels_plus, dtype='str', delimiter='\t')
        print('Shape of inputs, labels before transformation:', np.shape(inputs), np.shape(labels))

        inputs, labels = transform_input(inputs, labels)
        inputs = np.array(inputs)
        labels = np.array(labels)
        scaler = MinMaxScaler(feature_range=(0, 1))
        labels = scaler.fit_transform(labels)
        print('Shape of inputs, labels after transformation:', np.shape(inputs), np.shape(labels))

        inputs_4 = np.concatenate((inputs_4, inputs), axis=0)
        labels_4 = np.concatenate((labels_4, labels), axis=0)

x_test_2 = np.concatenate((inputs_3, inputs_4), axis=0)
%reset_selective -f "^inputs_3$"
%reset_selective -f "^inputs_4$"

y_test_2 = np.concatenate((labels_3, labels_4), axis=0)
%reset_selective -f "^labels_3$"
%reset_selective -f "^labels_4$"

x_test = np.concatenate((x_test_2, x_test_1), axis=0)
%reset_selective -f "^x_test_1$"
%reset_selective -f "^x_test_2$"

y_test = np.concatenate((y_test_2, y_test_1), axis=0)
%reset_selective -f "^y_test_1$"
%reset_selective -f "^y_test_2$"

#input_shape = x_train.shape[1:]
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

results = model.evaluate(x_test, y_test)
metrics = dict(zip(model.metrics_names, results))

