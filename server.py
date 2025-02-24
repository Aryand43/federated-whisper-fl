import flwr as fl  # Flower - Framework for Federated Learning
import torch
from typing import List, Tuple

# Custom function to aggregate weights from all clients using Federated Averaging (FedAvg)
def aggregate(weights: List[Tuple[List[torch.Tensor], int]]) -> List[torch.Tensor]:
    """
    Aggregates model weights using Federated Averaging (FedAvg).
    
    Args:
        weights: List of tuples (model_weights, num_samples) from clients.

    Returns:
        Aggregated model weights.
    """
    total_samples = sum(sample_count for _, sample_count in weights)  # Total number of data points across all clients

    # Compute weighted average of model parameters
    weighted_avg = [
        sum(weight * sample_count for weight, sample_count in weights) / total_samples
        for weight in zip(*[w for w, _ in weights])
    ]

    return weighted_avg  # Return aggregated model weights

# Define the Custom Federated Averaging Strategy
class WhisperFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        """
        Custom aggregation function for training.

        Args:
            rnd (int): Current training round.
            results (List): List of client training results (parameters, num_samples, etc.).
            failures (List): List of failed client updates.

        Returns:
            Aggregated parameters for the next round.
        """
        weights = [(parameters, num_examples) for parameters, num_examples, _ in results]  # Extract client weights
        aggregated_weights = aggregate(weights)  # Compute Federated Averaged weights
        return aggregated_weights, {}  # Return updated model parameters

# Start the Federated Learning Server
fl.server.start_server(
    server_address="0.0.0.0:8080",  # Server listens on port 8080
    config=fl.server.ServerConfig(num_rounds=5),  # Define number of FL training rounds
    strategy=WhisperFedAvg()  # Use our custom FedAvg strategy
)
