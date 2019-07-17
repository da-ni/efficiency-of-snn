import matplotlib.pyplot as plt
import keras
import numpy as np
from datetime import datetime

from config import *

now = datetime.now().strftime("%H-%M-%S")


def print_model_accuracy(model_history, filename=None):
    fig = plt.figure()
    plt.style.use('ggplot')
    plt.plot(model_history.history['acc'])
    plt.plot(model_history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    if bool(filename):
        if not filename.endswith(".png"):
            filename = filename + ".png"
        fig.savefig(filename, dpi=fig.dpi)
    else:
        plt.show()


def print_model_loss(model_history, filename=None):
    fig = plt.figure()
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    if bool(filename):
        if not filename.endswith(".png"):
            filename = filename + ".png"
        fig.savefig(filename, dpi=fig.dpi)
        print('saved figure as {}'.format(filename))
    else:
        plt.show()


def save_model(mdl, mdl_name, filename=None):
    if filename is None:
        fmt_str = "../models/ANN/{data}/{model}{version}_{time}.h5"
        filename = fmt_str.format(data=DATASET,
                                  model=mdl_name,
                                  version=VERSION,
                                  time=now)
    mdl.save(filename)
    print("Saved model as {}".format(filename))


def load_model(filename):
    mdl = keras.models.load_model(filename)
    # mdl.summary()
    return mdl
