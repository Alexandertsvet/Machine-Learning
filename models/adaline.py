import numpy as np


class AdalineGD:
    """
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=42):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += (self.eta*X.T).dot(errors)
            self.w_[0] += (self.eta*errors).sum()
            cost = (errors**2).sum()/2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """
        """
        return X

    def predict(self, X):
        """
        """
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


class AdalineSGD(object):
    """Классификатор на основе адаптивного линейного нейрона.
    Параметры:
        eta : float0
            Cкорость обучения (0.0 - 1.0)
        n_iter : int
            Колличество итерация по обучающему набору данных.
        shufle : bool (по умолчанию True)
            Перетасовка обучающих данных.
        random_state : генератор псевдослучайными числел.
            w_initiaizer : bool
                инициализация весов.
    Атрибуты:
        w_ : numpy.array([])
            Одномерный массив numpy весов после их
            инициализации случайными числами.
        cost_ : list()
            Значение функции издержки на основании суммы квадратов,
            усредненное по всем обучающим образцам в каждой эпохе.
    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter=n_iter
        self.w_initiaizer=False
        self.shuffle=shuffle
        self.random_state=random_state

    def fit(self, X, y) -> object:
        """Обучение.
        Параметры:
            X : обучающий массив
            y : целевой признак
        """
        self._initiaizer_weigths(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = _shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """
        Динамическое обучение, без повторной инициализации весов.
        """
        if not self.w_initiaizer:
            self._initiaizer_weigths(X.shape[1])
        if y.ravel().shape > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """
        Перемешивание обучающих данных.
        """
        _index = self.rgen.permutation(len(y))
        return X[_index], y[_index]

    def _initiaizer_weigths(self, m):
        """
        Инициализацтя весов небольшими случайными значениями
        Параметры:
            m : int
                m = X.shape[1]
                колличество признаков
        """
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initiaizer = True

    def _update_weights(self, xi, target):
        """
        Обновление весов.
        """
        output = self.activaion(self.net_input)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        """
        Фнкция общеего входа.
        """
        return np.dot(X, self.w_[1:])+self.w_[0]

    def activaion(self, X):
        """
        Линейная функция активации.
        """
        return X

    def predict(self, X):
        return np.where(self.activaion(self.net_input(X)) >= 0.0, 1, -1)
