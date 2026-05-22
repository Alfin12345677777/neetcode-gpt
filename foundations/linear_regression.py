import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places

      
        y = np.dot(X,weights)
        return np.round(y, 5)
        pass

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places

        
        difference = model_prediction - ground_truth 
        squared_error = difference ** 2
        loss = np.mean(squared_error)

        return round(loss,5)


        pass
