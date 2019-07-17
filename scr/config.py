import keras
from collections import OrderedDict

# Session Parameter
RANDOM_SEED = 45
VERSION = 'v1'


# Parameter for Training and Testing
N_EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 0.001

LOSS = keras.losses.categorical_crossentropy
OPTIMIZER = keras.optimizers.Adam(lr=LEARNING_RATE)
METRICS = 'accuracy'


# Dataset Specific Parameter
DATASET = 'cifar10'  # choose 'mnist' or 'cifar10'
N_CLASSES = 10


if DATASET == 'mnist':
    # Small CNN Parameter
    small = OrderedDict(
        NAME='SmallCnn',
        KERNEL=3,
        STRIDE=2,
        CONV1=32,
        DROPOUT_CNN=0.25,
        FLATTEN=True,
        FC1=64,
        DROPOUT=0.50,
    )

    # Medium CNN Parameter
    medium = OrderedDict(
        NAME='MediumCNN',
        KERNEL=3,
        STRIDE=2,
        CONV1_PADDING=32,
        DROPOUT_CNN_2=0.25,
        FLATTEN=True,
        FC1=64,
        DROPOUT=0.50,
    )

    # Large CNN Parameter
    large = OrderedDict(
        NAME='LargeCNN',
        KERNEL=3,
        STRIDE=2,
        CONV1_PADDING=16,
        DROPOUT_CNN_1=0.25,
        CONV2_PADDING=32,
        DROPOUT_CNN_2=0.25,
        FLATTEN=True,
        FC1=64,
        DROPOUT=0.50,
    )

    # Deep CNN Parameter
    deep = OrderedDict(
        NAME='DeepCNN',
        KERNEL=3,
        STRIDE=2,
        CONV1_PADDING=16,
        DROPOUT_CNN_1=0.25,
        CONV2_PADDING=32,
        DROPOUT_CNN_2=0.25,
        FLATTEN=True,
        FC1=64,
        FC2=32,
        FC3=16,
        DROPOUT=0.35
    )

else:
    # Small CNN Parameter
    small = OrderedDict(
        NAME='SmallCNN',
        KERNEL=3,
        STRIDE=2,
        CONV1=64,
        DROPOUT_CNN_2=0.25,
        FLATTEN=True,
        FC1=256,
        DROPOUT=0.50,
    )

    # Medium CNN Parameter
    medium = OrderedDict(
        NAME='MediumCNN',
        KERNEL=3,
        STRIDE=2,
        CONV2_PADDING=64,
        DROPOUT_CNN_2=0.25,
        FLATTEN=True,
        FC1=256,
        DROPOUT=0.50,
    )

    # Large CNN Parameter
    large = OrderedDict(
        NAME='LargeCNN',
        KERNEL=3,
        STRIDE=2,
        CONV1_PADDING=32,
        DROPOUT_CNN_1=0.25,
        CONV2_PADDING=64,
        DROPOUT_CNN_2=0.25,
        FLATTEN=True,
        FC1=512,
        DROPOUT=0.50,
    )

    # Deep CNN Parameter
    deep = OrderedDict(
        NAME='DeepCNN',
        KERNEL=3,
        STRIDE=2,
        CONV1_PADDING=32,
        DROPOUT_CNN_1=0.25,
        CONV2_PADDING=64,
        DROPOUT_CNN_2=0.25,
        FLATTEN=True,
        FC1=512,
        FC2=256,
        FC3=128,
        DROPOUT=0.35,
    )

# choose which parameter dicts should be used
CHOSEN_DICTS = [small, medium, large, deep]

