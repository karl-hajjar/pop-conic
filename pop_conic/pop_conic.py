import numpy as np
from tqdm import tqdm


INPUT_DIM = 50
BIAS = False

SEED = 42
N_TRAIN = 256
BATCH_SIZE = 64
ALPHA = 1e-1
BASE_LR = 1.0e-2
N_STEPS = int(1e3)
N_VAL = 100
VAL_ITER = 100


def forward(a, b, x):
    return np.matmul(np.maximum(np.matmul(x, b.T), 0), a.T)


def relu_prime(z):
    return (np.sign(z) + 1) / 2


def sample_neurons(n_samples=1):
    return np.random.normal(size=(n_samples, INPUT_DIM))


def forward_states(a, b, x):
    h_1 = np.matmul(x, b.T)
    x_1 = np.maximum(h_1, 0)
    y_hat = np.matmul(x_1, a.T)

    return h_1, x_1, y_hat


def initialize_neurons(x, y, n_neurons=1):
    # empty arrays and null prediction function
    a = np.array([[]])
    b = np.array([[]])
    y_hat = 0

    b_hat = sample_neurons(n_samples=n_neurons)
    x_1 = np.maximum(np.matmul(x, b_hat.T), 0)
    da = (y - y_hat) * x_1
    a_hat = ALPHA * np.mean(da, axis=0)

    a = np.append(a, a_hat).reshape(1, n_neurons)
    b = np.append(b, b_hat).reshape(n_neurons, -1)

    return a, b


def update(a, b, x, y, n_updates=1, compute_loss=True):
    # print('a.shape', a.shape)
    # print('b.shape', b.shape)
    for _ in range(n_updates):
        h_1, x_1, y_hat = forward_states(a, b, x)
        delta = y_hat - y
        dh_1 = relu_prime(h_1)
        # print('delta.shape', delta.shape)
        # print('x_1.shape', x_1.shape)
        # print('dh_1.shape', dh_1.shape)
        # print('np.diag(a.reshape(-1)).shape', np.diag(a.reshape(-1)).shape)
        dh = np.matmul(relu_prime(h_1), np.diag(a.reshape(-1))).T
        # print('dh.shape', dh.shape)
        b = b - BASE_LR * np.matmul(dh,
                                    delta * x_train) / N_TRAIN
        a = a - BASE_LR * np.mean(delta * x_1, axis=0)

        loss = -1.0
        if compute_loss:
            loss = 0.5 * np.mean(delta ** 2)

    return a, b, loss


def initialize_net(x, y, n_neurons=1):
    # initialize neurons  y sampling and setting weights accordingly
    a, b = initialize_neurons(x, y, n_neurons=n_neurons)
    # print(a.shape)
    # print(b.shape)
    print(a)
    print(b)

    # return as initialization for the network the first update for the sampled neurons / weights
    return update(a, b, x, y, n_updates=1)


def train(n_steps, a, b, x_train, y_train, x_val, y_val, do_val=True, n_samples=1, n_updates=1):
    train_losses = []
    val_losses = []

    for i in tqdm(range(n_steps)):
        # sample neurons and set weights
        if n_samples > 0:
            b_hat = sample_neurons(n_samples=n_samples)
            x_1 = np.maximum(np.matmul(x_train, b_hat.T), 0)
            a_hat = ALPHA * np.mean((y_train - y_hat).reshape(-1, 1) * x_1, axis=0)

            # add newly sampled neurons / weights to the list
            a = np.append(a, a_hat)
            b = np.append(b, b_hat)

        # update neurons / weights
        a, b, train_loss = update(a, b, x_train, y_train, n_updates=n_updates)
        train_losses.append(train_loss)

        # validation loop if do_val is True
        if do_val:
            if i % VAL_ITER == 0:
                y_hat = forward(a, b, x_val)
                val_loss = 0.5 * np.mean((y_hat - y_val) ** 2)
                val_losses.append(val_loss)
                print('Train loss at step {:,} : {:.5f}'.format(i, train_loss))
                print('Validation loss at step {:,} : {:.5f}'.format(i, val_loss))

    return a, b, np.array(train_losses), np.array(val_losses)
