"""
mixdenoise.py: Implementation of MixDenoise module.

MixDenoise is a learnable module that separates an input sample
into corresponding probabilities from a series of learned gaussians.

This module is useful if you have a value `x` coming from an unknown
mixture of gaussians and want to split it into its components.

(TODO: better description here)
"""
import math
import torch
import torch.nn as nn
from torch.autograd import Function

class GaussianProb(Function):
    """
    Implementation of a gaussian probability estimation function with custom backwards gradient-passing.
    """
    @staticmethod
    def forward(ctx, sample, means, stdvs):
        """Forward pass: compute a vector of gaussian probabilities."""
        return None #TODO

    @staticmethod
    def backward(ctx, grad_output):
        """Backwards pass: derivative of gaussian density function w/r/t sample."""
        return None #TODO


class MixDenoise(nn.Module):
    """
    (...)
    """
    def __init__(self, ncomponents, eps=0.001):
        """
        (...)
        """
        super(MixDenoise, self).__init__()
        self.ncomponents = ncomponents
        self.eps = eps
        # initialize means and stdvs:
        self.means = nn.Parameter(torch.randn(ncomponents).mul(0.01), requires_grad=True)
        self.stdvs = nn.Parameter(torch.rand(ncomponents).mul(0.01).add(1.0), requires_grad=True)

    def forward(self, x):
        """
        For each component, compute the density function.

        Args:
        * x: a FloatTensor variable of shape (batch_size, 1); presumably sampled from a
        mixture of gaussians.

        Returns:
        * a FloatTensor variable of shape (batch_size, ncomponents).

        [[TODO: factor this out into an `autograd.Function` and implement below.]]
        """
        # expand xd to appropriate shape:
        xd = x.expand(x.size(0), self.ncomponents)

        # compute densities as batch:
        variances = torch.pow(self.stdvs, 2).clamp(min=self.eps)
        exponent = torch.exp(torch.pow(xd - self.means, 2).mul(-1.0) / (2 * variances))
        multiplier = torch.reciprocal(torch.sqrt(variances.mul(2*math.pi)))
        return (multiplier * exponent)
