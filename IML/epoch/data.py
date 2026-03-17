import numpy as np

def load_data():
    X = np.random.rand(100, 1)
    y = 3 * X + 2
    return X, y
