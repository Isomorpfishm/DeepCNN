import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, Activation, Input, Cropping1D, Flatten, MaxPooling1D
from tensorflow.keras.models import Model
from keras.optimizers import Adam

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, MinMaxScaler

import os
import time

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
            y = tf.keras.layers.add([Conv1D(32, kernel_size=1, strides=1, padding='same', dilation_rate=1)(x), y])

    x = Conv1D(32, kernel_size=1, strides=1, padding='same', dilation_rate=1)(x)
    # adding up with what was shortcut from the prev layers
    x = tf.keras.layers.add([x, y])

    x = Conv1D(1, kernel_size=1, strides=1, padding='same', dilation_rate=1)(x)
    x = Flatten()(x)
    outputs = Dense(1, activation='relu')(x)

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

chr_1 = ['chr2', 'chr4', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10']
chr_2 = ['chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17']
chr_3 = ['chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY'] 
chr_all = chr_1 + chr_2 + chr_3

# <----------------- Plus strand ------------------>
dir_inputs = './gdrive/MyDrive/dataset-new/seq/plus/'
dir_labels = './gdrive/MyDrive/dataset-new/seq/plus/'

for chrN in chr_all:
    
    filedir_input_minus = os.path.join(dir_inputs, chrN +'.txt')
    filedir_labels_minus = os.path.join(dir_labels, chrN +'-scores.txt')

    # import, process, concatenate with prev files
    if chrN == 'chr2':
        inputs_1 = np.loadtxt(filedir_input_minus, dtype='str', delimiter='\t')
        labels_1 = np.loadtxt(filedir_labels_minus, dtype='str', delimiter='\t')
        print('Shape of inputs, labels before transformation:', np.shape(inputs_1), np.shape(labels_1))

        inputs_1, labels_1 = transform_input(inputs_1, labels_1)
        inputs_1 = np.array(inputs_1)
        labels_1 = np.array(labels_1)
        print('Shape of inputs, labels after transformation:', np.shape(inputs_1), np.shape(labels_1))

    else:
        inputs = np.loadtxt(filedir_input_minus, dtype='str', delimiter='\t')
        labels = np.loadtxt(filedir_labels_minus, dtype='str', delimiter='\t')
        print('Shape of inputs, labels before transformation:', np.shape(inputs), np.shape(labels))

        inputs, labels = transform_input(inputs, labels)
        inputs = np.array(inputs)
        labels = np.array(labels)
        #scaler = MinMaxScaler(feature_range=(0, 1))
        #labels = scaler.fit_transform(labels)
        print('Shape of inputs, labels after transformation:', np.shape(inputs), np.shape(labels))

        inputs_1 = np.concatenate((inputs_1, inputs), axis=0)
        labels_1 = np.concatenate((labels_1, labels), axis=0)


# <----------------- Minus strand ------------------>
dir_inputs = './gdrive/MyDrive/dataset-new/seq/minus/'
dir_labels = './gdrive/MyDrive/dataset-new/seq/minus/'

for chrN in chr_all:
    
    filedir_input_minus = os.path.join(dir_inputs, chrN +'.txt')
    filedir_labels_minus = os.path.join(dir_labels, chrN +'-scores.txt')

    # import, process, concatenate with prev files
    if chrN == 'chr2':
        inputs_2 = np.loadtxt(filedir_input_minus, dtype='str', delimiter='\t')
        labels_2 = np.loadtxt(filedir_labels_minus, dtype='str', delimiter='\t')
        print('Shape of inputs, labels before transformation:', np.shape(inputs_2), np.shape(labels_2))

        inputs_2, labels_2 = transform_input(inputs_2, labels_2)
        inputs_2 = np.array(inputs_2)
        labels_2 = np.array(labels_2)
        print('Shape of inputs, labels after transformation:', np.shape(inputs_2), np.shape(labels_2))

    else:
        inputs = np.loadtxt(filedir_input_minus, dtype='str', delimiter='\t')
        labels = np.loadtxt(filedir_labels_minus, dtype='str', delimiter='\t')
        print('Shape of inputs, labels before transformation:', np.shape(inputs), np.shape(labels))

        inputs, labels = transform_input(inputs, labels)
        inputs = np.array(inputs)
        labels = np.array(labels)
        #scaler = MinMaxScaler(feature_range=(0, 1))
        #labels = scaler.fit_transform(labels)
        print('Shape of inputs, labels after transformation:', np.shape(inputs), np.shape(labels))

        inputs_2 = np.concatenate((inputs_2, inputs), axis=0)
        labels_2 = np.concatenate((labels_2, labels), axis=0)

x_train = np.concatenate((inputs_1, inputs_2), axis=0)
%reset_selective -f "^inputs_1$"
%reset_selective -f "^inputs_2$"

y_train = np.concatenate((labels_1, labels_2), axis=0)
%reset_selective -f "^labels_1$"
%reset_selective -f "^labels_2$"

scaler = MinMaxScaler(feature_range=(0, 1))
y_train = scaler.fit_transform(y_train)

input_shape = x_train.shape[1:]
print('x_train shape:', x_train.shape)
print(y_train.shape[0], 'train samples')
print('y_train shape:', y_train.shape)

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "./gdrive/MyDrive/model/training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model = model_CNN_full(input_shape=input_shape, rb=3, dil=6, kernel_size=6)

model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              metrics=tf.keras.metrics.MeanSquaredError())

print(model.summary())

# Save the weights using the `checkpoint_path` format
model.load_weights('./gdrive/MyDrive/model/model-regression-onelabel-classparams.h5')

# add layers
x = Conv1D(32, kernel_size=1, strides=1, padding='same', dilation_rate=1)(model.layers[-4].output)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
x = Conv1D(32, kernel_size=1, strides=1, padding='same', dilation_rate=1)(x)
x = BatchNormalization(axis=-1)(x)
x = Conv1D(1, kernel_size=1, strides=1, padding='same', dilation_rate=1)(x)
x = keras.layers.Flatten()(x)
o = Dense(1, activation='relu')(x)


# take output after last addition layer and before Conv1D(1)
# constructing a model with the same inputs but a diff output
model2 = Model(inputs=model.input, outputs=[o])

# copy the weights
for l_tg,l_sr in zip(model2.layers[:-4], model.layers[:-4]):
    wk0=l_sr.get_weights()
    l_tg.set_weights(wk0)

model2.summary()

# compile model2 and train for 1-2 epochs
model2.compile(loss=tf.keras.losses.MeanSquaredError(),
               optimizer=Adam(learning_rate=1e-5),
               metrics=tf.keras.metrics.MeanSquaredError())

model2.save_weights(checkpoint_path.format(epoch=0))

history = model2.fit(x_train, y_train, epochs=3, batch_size=70, shuffle=True, validation_split = 0.2)

model2.save('./gdrive/MyDrive/model/model-finetune-unfrozen.h5')



chr = ['chr1', 'chr3', 'chr5']

# <------------------ Plus strand ------------------>
dir_inputs = './gdrive/MyDrive/dataset-new/seq/plus/'
dir_labels = './gdrive/MyDrive/dataset-new/seq/plus/'

for chrN in chr:
    
    filedir_input_minus = os.path.join(dir_inputs, chrN +'.txt')
    filedir_labels_minus = os.path.join(dir_labels, chrN +'-scores.txt')

    # import, process, concatenate with prev files
    if chrN == 'chr1':
        inputs_1 = np.loadtxt(filedir_input_minus, dtype='str', delimiter='\t')
        labels_1 = np.loadtxt(filedir_labels_minus, dtype='str', delimiter='\t')
        print('Shape of inputs, labels before transformation:', np.shape(inputs_1), np.shape(labels_1))

        inputs_1, labels_1 = transform_input(inputs_1, labels_1)
        inputs_1 = np.array(inputs_1)
        labels_1 = np.array(labels_1)
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




# <----------------- Minus strand ------------------->
dir_inputs = './gdrive/MyDrive/dataset-new/seq/minus/'
dir_labels = './gdrive/MyDrive/dataset-new/seq/minus/'

for chrN in chr:
    filedir_input_minus = os.path.join(dir_inputs, chrN +'.txt')
    filedir_labels_minus = os.path.join(dir_labels, chrN +'-scores.txt')

    # import, process, concatenate with prev files
    if chrN == 'chr1':
        inputs_2 = np.loadtxt(filedir_input_minus, dtype='str', delimiter='\t')
        labels_2 = np.loadtxt(filedir_labels_minus, dtype='str', delimiter='\t')
        print('Shape of inputs, labels before transformation:', np.shape(inputs_2), np.shape(labels_2))

        inputs_2, labels_2 = transform_input(inputs_2, labels_2)
        inputs_2 = np.array(inputs_2)
        labels_2 = np.array(labels_2)
        print('Shape of inputs, labels after transformation:', np.shape(inputs_2), np.shape(labels_2))

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

        inputs_2 = np.concatenate((inputs_2, inputs), axis=0)
        labels_2 = np.concatenate((labels_2, labels), axis=0)

x_test = np.concatenate((inputs_1, inputs_2), axis=0)
%reset_selective -f "^inputs_1$"
%reset_selective -f "^inputs_2$"

y_test = np.concatenate((labels_1, labels_2), axis=0)
%reset_selective -f "^labels_1$"
%reset_selective -f "^labels_2$"

scaler = MinMaxScaler(feature_range=(0, 1))
y_test = scaler.fit_transform(y_test)

print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

results = model2.evaluate(x_test, y_test)
metrics = dict(zip(model2.metrics_names, results))
