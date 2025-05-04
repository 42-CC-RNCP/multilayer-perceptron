from litetorch.loss.binary_cross_entropy import BinaryCrossEntropyLoss
from litetorch.loss.cross_entropy import CrossEntropyLoss
from litetorch.loss.mse import MSELoss
from litetorch.optim.SGD import SGD
from litetorch.nn.activation import *


LOSS_REGISTRY = {
    "binary_cross_entropy": BinaryCrossEntropyLoss,
    "cross_entropy": CrossEntropyLoss,
    "mse": MSELoss,
}

OPTIMIZER_REGISTRY = {
    "sgd": SGD,
}

ACTICATION_REGISTRY = {
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "softmax": Softmax,
    "leaky_relu": LeakyReLU,
}
