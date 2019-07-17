from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from config import *


input_shape = None


class ConvNet:

    def __init__(self, param_dict):
        # initialize as sequential model
        self.model = Sequential()

        self.name = param_dict.pop('NAME')
        self.kernel = param_dict.pop('KERNEL')
        self.stride = param_dict.pop('STRIDE')

        # add input layer (pop first item of dict)
        key, value = param_dict.popitem(last=False)
        self.add_first_layer(key, value)

        # add predefined layer architecture
        for key, value in param_dict.items():
            self.add_layer(key, value)

        # add output layer
        self.model.add(Dense(N_CLASSES, activation='softmax'))

        # compile the model
        self.model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=[METRICS])
        print("\nInitialized "+str(self.name))
        self.model.summary()

    def add_first_layer(self, key, value):
        global input_shape
        assert "CONV" in key
        if 'PADDING' in key:
            self.model.add(Conv2D(value, kernel_size=self.kernel, padding='same',
                                  input_shape=input_shape, activation='relu'))
            self.model.add(Conv2D(value, kernel_size=self.kernel, activation='relu'))
        else:
            self.model.add(Conv2D(value, kernel_size=self.kernel,
                                  input_shape=input_shape, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=self.stride))

    def add_layer(self, key, value):
        if "CONV" in key:
            if "PADDING" in key:
                self.model.add(Conv2D(value, kernel_size=self.kernel, padding='same', activation='relu'))
            self.model.add(Conv2D(value, kernel_size=self.kernel, activation='relu'))
            self.model.add(MaxPooling2D(pool_size=self.stride))
        elif "DROPOUT_CNN" in key:
            self.model.add(Dropout(value))
        elif "FLATTEN" in key:
            self.model.add(Flatten())
        elif "FC" in key:
            self.model.add(Dense(value, activation='relu'))
        elif "DROPOUT" == key:
            self.model.add(Dropout(value))
        else:
            raise ValueError('Layer of kind {} currently not supported.'.format(key))


def load_preprocessed_data(data='mnist', show_prints=True, save_data=False):
    global input_shape
    assert data == 'mnist' or data == 'cifar10'
    if data == 'mnist':
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        input_shape = 1, *x_train.shape[1:]
    else:
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        input_shape = x_train.shape[1:]

    # reshape and normalize x_train and x_test
    x_train = x_train.reshape(x_train.shape[0], *input_shape)
    x_test = x_test.reshape(x_test.shape[0], *input_shape)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, N_CLASSES)
    y_test = keras.utils.to_categorical(y_test, N_CLASSES)

    if show_prints:
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

    if save_data:
        np.savez('x_norm', x_train[:2000, ...])
        np.savez('x_test', x_test)
        np.savez('y_test', y_test)
    return x_train, y_train, x_test, y_test

