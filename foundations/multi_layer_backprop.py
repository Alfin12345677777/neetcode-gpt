import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        x = np.array(x, dtype=float)
        w1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        w2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)
        

        z1 = w1 @ x + b1 
        a1 = np.maximum(0,z1)
        z2 = w2 @ a1 + b2
        y_pred = z2
        loss = np.mean((y_pred - y_true)**2)

        n = len(y_true)
        idk = 2/n
        dz2 = idk * (y_pred - y_true)

        dw2 = np.outer(dz2, a1)
        
        db2 = dz2

        da1 = w2.T @ dz2



        relu_mask = (z1 > 0).astype(float)
        dz1 = da1 * relu_mask

        db1 = dz1



        
        dw1 =np.outer(dz1, x)


        return {
            "loss": float(round(loss,4)),
            "dW1": (np.round(dw1 + 0.0, 4)).tolist(),
            "db1": (np.round(db1 + 0.0, 4)).tolist(),
            "dW2": (np.round(dw2 + 0.0, 4)).tolist(),
            "db2": (np.round(db2 + 0.0, 4)).tolist()
        }
