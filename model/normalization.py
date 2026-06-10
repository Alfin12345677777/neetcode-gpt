import numpy as np
from numpy.typing import NDArray



class Solution:
    def forward(self, x, gamma, beta):
        x_hat = (x - x.mean()) / np.sqrt(x.var() + 1e-5)
        return np.round(gamma * x_hat + beta, 5)