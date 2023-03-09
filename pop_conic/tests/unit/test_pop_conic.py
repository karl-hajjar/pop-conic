import unittest
from pop_conic import *
from utils.tools import set_random_seeds


class TestPopConic(unittest.TestCase):
    def setUp(self) -> None:
        set_random_seeds(SEED)

        # Teacher Network
        m = 50  # number of teacher neurons
        eps = 3e-1

        a_star = 10 * np.sign(np.random.uniform(size=(1, m)) - 0.5) / m
        b_star = 2 * np.random.uniform(size=(m, INPUT_DIM)) - 1 - eps * np.random.normal(size=(m, INPUT_DIM)) ** 2

        # Data
        self.x_train = np.random.normal(size=(N_TRAIN, INPUT_DIM))
        self.y_train = forward(a_star, b_star, self.x_train)

        self.x_val = np.random.normal(size=(N_VAL, INPUT_DIM))
        self.y_val = forward(a_star, b_star, self.x_val)

    def test_init_net(self):
        n_neurons = 100
        n_updates = 1

        a, b, _ = initialize_net(self.x_train, self.y_train, n_neurons=n_neurons)

    def test_pop_conic(self):
        # FULL LOOP (POP-CONIC 1 sample & 1 update per step)
        n_neurons = 1
        n_updates = 1

        a, b, _ = initialize_net(self.x_train, self.y_train, n_neurons=n_neurons)
        a, b, train_losses, val_losses = train(N_STEPS, a, b, self.x_train, self.y_train, self.x_val, self.y_val,
                                               n_samples=n_neurons, n_updates=n_updates)


if __name__ == '__main__':
    unittest.main()
