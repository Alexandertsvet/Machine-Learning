import numpy as np


class Perceptron(object):
    """
    Классификация на основе перцептрона.
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=42):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        Обучение.
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta*(target-self.predict(xi))
                self.w_[1:] += update*xi
                self.w_[0] += update
                errors += int(update != 0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """
        Вычисление общего входа.
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """
        Возврашает метку класса после одной итерации
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
