from litetorch.nn.loss import *
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
