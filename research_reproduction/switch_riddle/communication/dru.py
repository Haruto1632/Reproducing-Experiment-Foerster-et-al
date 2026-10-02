"""
Discretise / Regularise Unit (DRU)
===================================
Foerster et al. 2016, Sec 5.2  [A]

During TRAINING (regularise mode):
    DRU(m) = Logistic( N(m, σ) )
    where σ = 2  [A]

During EXECUTION / EVAL (discretise mode):
    DRU(m) = 1{ m > 0 }
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DRU(nn.Module):
    """
    Discretise/Regularise Unit.

    Parameters
    ----------
    sigma : float
        Standard deviation of Gaussian noise added during training.
        Paper specifies σ = 2.                                        [A]
    """

    def __init__(self, sigma: float = 2.0):
        super().__init__()
        self.sigma = sigma                                           # [A]

    def forward(self, m: torch.Tensor, training: bool) -> torch.Tensor:
        """
        Parameters
        ----------
        m        : raw message tensor, any shape
        training : if True, apply regularise mode; else discretise mode

        Returns
        -------
        Processed message, same shape as m.
        """
        if training:
            # Regularise: Logistic( N(m, σ) )                       [A]
            noise = torch.randn_like(m) * self.sigma
            return torch.sigmoid(m + noise)
        else:
            # Discretise: 1{m > 0}                                  [A]
            return (m > 0).float()
