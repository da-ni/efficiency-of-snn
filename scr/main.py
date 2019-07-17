from tensorflow import set_random_seed
from keras.callbacks import TensorBoard

from models import *
from utils import *


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    set_random_seed(RANDOM_SEED)

    # list of parameter dicts we want use
    param_dicts = CHOSEN_DICTS

    # load training data
    X_t, y_t, X_val, y_val = load_preprocessed_data(DATASET)

    for params in param_dicts:
        # initialize model
        net = ConvNet(params)
        model = net.model
        print('\nTraining of {} ...\n'.format(net.name))

        # set-up for tensorboard
        fmt = "../models/test"
        log_dir = fmt.format(data=DATASET,
                             model=net.name,
                             version=VERSION,
                             time=now)
        tensorboard_callback = TensorBoard(log_dir=log_dir)

        # train model (history is used for accuracy/loss plots)
        history = model.fit(X_t, y_t,
                            batch_size=BATCH_SIZE,
                            epochs=N_EPOCHS,
                            verbose=2,
                            validation_data=(X_val, y_val),
                            callbacks=[tensorboard_callback],
                            )

        # save_model(model, net.name)

        # print score to console
        score = model.evaluate(X_val, y_val, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        # # save figures
        # ffmt = '../figures/{}_{}{}_{}'
        # print_model_accuracy(history, filename=ffmt.format(DATASET, net.name, VERSION, 'accuracy'))
        # print_model_loss(history, filename=ffmt.format(DATASET, net.name, VERSION, 'loss'))