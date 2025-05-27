import torch
import torch.nn as nn
import torch.nn.functional as F

# === Custom similarity loss ===
def combined_loss(delta_pred, delta_true, recon_pred, recon_true, alpha=0.5, beta=0.5):
    # Similarity loss (cos + norm)
    delta_pred_normed = F.normalize(delta_pred, dim=1)
    delta_true_normed = F.normalize(delta_true, dim=1)
    cos_loss = 1 - torch.sum(delta_pred_normed * delta_true_normed, dim=1).mean()

    norm_pred = delta_pred.norm(dim=1)
    norm_true = delta_true.norm(dim=1)
    norm_ratio = torch.minimum(norm_pred, norm_true) / torch.maximum(norm_pred, norm_true + 1e-8)
    norm_loss = 1 - norm_ratio.mean()

    similarity_loss = alpha * cos_loss + (1 - alpha) * norm_loss

    # Reconstruction loss
    recon_loss = F.mse_loss(recon_pred, recon_true)

    # Total
    total = beta * similarity_loss + (1 - beta) * recon_loss
    return total, cos_loss.item(), norm_loss.item(), recon_loss.item()

# === MLP ===
class DeltaPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], output_dim)
        )

    def forward(self, x):
        return self.layers(x)