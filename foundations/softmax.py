import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)

        z_stable = z - np.max(z)

        z_max = np.exp(z_stable)


        
        softmax = z_max/np.sum(z_max)

        return np.round(softmax,4)
        pass
