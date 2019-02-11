import numpy
from tensorflow.keras import callbacks
from tensorflow.python.keras import backend as K


class LearningRateRangeTest(callbacks.Callback):
    def __init__(self, total_nb_steps, base_rate=10e-5, max_rate=10e0):
        self.max_rate = np.log10(max_rate)
        self.base_rate = np.log10(base_rate)
        self.total_nb_steps = float(total_nb_steps)
        self.steps_taken = 0
        super(LearningRateRangeTest, self).__init__()

    def on_train_begin(self, logs=None):
        K.set_value(self.model.optimizer.lr, 10 ** self.base_rate)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.steps_taken += 1
        _lr = 10 ** (self.base_rate * (1 - self.steps_taken / self.total_nb_steps))
        logs['lr'] = np.float64(_lr)
        K.set_value(self.model.optimizer.lr, np.float64(_lr))


class CosineLearninRatePolicy(callbacks.Callback):
    def __init__(self, max_rate, total_nb_steps):
        self.base_rate = max_rate / 10
        self.max_rate = max_rate
        self.steps_taken = 0
        self.cycle_length = float(total_nb_steps) / 2 # nb steps per epoch / 2
        super(CosineLearninRatePolicy, self).__init__()

    def on_train_begin(self, logs=None):
        K.set_value(self.model.optimizer.lr, self.max_rate)
        # for tensorboard
        logs['lr'] = np.float64(self.max_rate)

    def on_batch_end(self, batch, logs=None):
        self.steps_taken += 1
        _scaler = (1 + np.cos(np.pi * self.steps_taken / self.cycle_length)) / 2
        _lr = self.base_rate + (self.max_rate - self.base_rate) * _scaler
        # for tensorboard
        logs['lr'] = np.float64(_lr)
        K.set_value(self.model.optimizer.lr, _lr)