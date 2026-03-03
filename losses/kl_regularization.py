import torch
import torch.nn as nn
from torch.distributions import Dirichlet


class KLDivergenceLoss(nn.Module):
    """
    KL( Dir(alpha) || Dir(1) )
    """

    def __init__(self, num_classes):
        super(KLDivergenceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, alpha):
        """
        Args:
            alpha (torch.Tensor): Dirichlet parameters [N, K]

        Returns:
            torch.Tensor: scalar KL loss
        """

        prior = torch.ones_like(alpha)

        dir_p = Dirichlet(alpha)
        dir_q = Dirichlet(prior)

        kl = torch.distributions.kl_divergence(dir_p, dir_q)

        return torch.mean(kl)