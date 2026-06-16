import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)

        # Machine 1: turn 784 pixel values into 512 hidden features
        self.fc1 = nn.Linear(784, 512)

        # Machine 2: remove negative values
        self.relu = nn.ReLU()

        # Machine 3: randomly drop 20% of hidden features
        self.dropout = nn.Dropout(0.2)

        # Machine 4: turn 512 hidden features into 10 digit scores
        self.fc2 = nn.Linear(512,10)

        # Machine 5: squash scores into 0-1 confidence values
        self.sigmoid = nn.Sigmoid()

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places

        x = images

        # Step 1: pixels -> hidden features
        x = self.fc1(x)

        # Step 2: apply ReLU
        x = self.relu(x)

        # Step 3: apply dropout
        x = self.dropout(x)

        # Step 4: hidden features -> 10 outputs
        x = self.fc2(x)

        # Step 5: convert outputs to probabilities/confidences
        x = self.sigmoid(x)

        # Step 6: round to 4 decimals and return
        return torch.round(x, decimals=4)
        pass
