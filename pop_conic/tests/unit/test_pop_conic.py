import unittest
from pop_conic import *
from utils.tools import set_random_seeds


class TestPopConic(unittest.TestCase):
    def setUp(self) -> None:
        set_random_seeds(SEED)

        # Teacher Network
        m = 50  # number of teacher neurons
        eps = 1e-3

        a_star = 0.5 * np.sign(np.random.uniform(size=(1, m)) - 0.5) / m
        b_star = 2 * np.random.uniform(size=(m, INPUT_DIM)) - 1 + eps * np.random.normal(size=(m, INPUT_DIM)) ** 2

        # Data
        self.x_train = np.random.normal(size=(N_TRAIN, INPUT_DIM))
        self.y_train = forward(a_star, b_star, self.x_train)

        self.x_val = np.random.normal(size=(N_VAL, INPUT_DIM))
        self.y_val = forward(a_star, b_star, self.x_val)

    def test_init_net(self):
        n_neurons = 100
        n_updates = 1

        a, b, _ = initialize_net(self.x_train, self.y_train, n_neurons=n_neurons)


if __name__ == '__main__':
    unittest.main()
