import os
from glob import glob
from time import sleep

from numpy.random import seed
import tensorflow as tf
import keras.backend as K

from utils import load_model
from models import load_preprocessed_data
from config import RANDOM_SEED


def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.


def list_files(directory, file_extensions=('h5')):
    '''
    Returns a list of files (via DFS in given directory) with specified extensions.

    :param directory: str
    :param file_extensions: array(str)

    :return: filenames: Array(str)
    '''
    filenames = []
    dir_suffix = "/**/*"
    for file in glob(directory + dir_suffix):
        if os.path.isfile(file) and file.split(".")[-1] in file_extensions:
            filenames.append(file)
    return filenames


if __name__ == '__main__':
    seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)
    path = '../models/ANN'
    files = list_files(path)
    flops = []
    mnist_files = {file for file in files if 'mnist' in file}
    cifar_files = set(files) - mnist_files

    for f in reversed(files):
        model = load_model(f)
        flop = get_flops(model)
        flops.append(flop)

    for f, flop in zip(files, flops):
        model = load_model(f)
        print('\n', 'Examining Model: ', f.split('/')[-2:], '\n')
        model.summary()
        print('Total flops: {:,}'.format(flop))

    # X_t, y_t, X_val, y_val = load_preprocessed_data('mnist')
    # for f in mnist_files:
    #     model = load_model(f)
    #     score = model.evaluate(X_val, y_val, verbose=0)
    #     model.summary()
    #     print('Test loss:', score[0])
    #     print('Test accuracy:', score[1])
    #
    # print('\n now cifar files \n')
    # X_t, y_t, X_val, y_val = load_preprocessed_data('cifar10')
    # for f in cifar_files:
    #     model = load_model(f)
    #     score = model.evaluate(X_val, y_val, verbose=0)
    #     model.summary()
    #     print('Test loss:', score[0])
    #     print('Test accuracy:', score[1])


