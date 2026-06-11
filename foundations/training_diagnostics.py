import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []

        with torch.no_grad():
            current = x
            for layer in model:
                current = layer(current)

                if isinstance(layer, nn.Linear):
                    mean = round(current.mean().item(), 4)
                    std = round(current.std().item(), 4)
                    dead_neurons = (current <= 0).all(dim=0)
                    dead_fraction = dead_neurons.float().mean()
                    dead_fraction = round(dead_fraction.item(), 4)

                    stats.append({
                    "mean": mean,
                    "std": std,
                    "dead_fraction": dead_fraction
                })

        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.

        model.zero_grad()

        pred = model(x)

        loss_fn = nn.MSELoss()
        loss = loss_fn(pred, y)
        loss.backward()

        stats = []
       

        for layer in model:
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad

                stats.append({
                    "mean": round(grad.mean().item(), 4),
                    "std": round(grad.std().item(), 4),
                    "norm": round(torch.norm(grad).item(), 4)
                })

        return stats

        pass

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
            # 1. Dead neurons
        for stat in activation_stats:
            if stat["dead_fraction"] > 0.5:
                return "dead_neurons"

        # 2. Exploding gradients
        for stat in gradient_stats:
            if stat["norm"] > 1000:
                return "exploding_gradients"

        # 3. Vanishing gradient in last layer
        if gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        # 4. Vanishing activations
        for stat in activation_stats:
            if stat["std"] < 0.1:
                return "vanishing_gradients"

        # 5. Exploding activations
        for stat in activation_stats:
            if stat["std"] > 10.0:
                return "exploding_gradients"

        # 6. No problem found
        return "healthy"
