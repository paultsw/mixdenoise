"""
mixdenoise.py: Implementation of MixDenoise module.

MixDenoise is a learnable module that separates an input sample
into corresponding probabilities from a series of learned gaussians.

This module is useful if you have a value `x` coming from an unknown
mixture of gaussians and want to split it into its components.

Credits:
* Thanks to @hardmaru (https://github/com/hardmaru/pytorch_notebooks) for
providing insight through publicly available code implementing the MDN loss function.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MixDenoise(nn.Module):
    """
    The MixDenoise layer is essentially a container for the pi, mu, and sigma parameters
    of a gaussian mixture model, which are evolved by backpropagation via a mixture density
    loss function which optimizes the probability of a stream of inputs `x ~ GMM(unknown)`
    coming from the parameters in here.
    """
    def __init__(self, num_components, jitter=0.01):
        """
        Construct and initialize the parameters of a gaussian mixture model with specified
        number of components. Note that we initialize the parameters using typical non-informative
        priors: for K := num_components, we set the following initial probabilities:
        * mixture coefficients are set to 1.0 (+/- jitter);
        * mu values are set to 0.0 (+/- jitter);
        * sigma values are set to 1.0 (+/- jitter).
        """
        super(MixDenoise, self).__init__()

        # save input arguments for future reference:
        self.num_components = num_components
        self.jitter = jitter
        
        # construct un-normalized model parameters:
        _mixes = torch.ones(num_components).add_(torch.randn(num_components).mul_(jitter))
        _means = torch.zeros(num_components).add_(torch.randn(num_components).mul_(jitter))
        _stdvs = torch.ones(num_components).add_(torch.randn(num_components).mul_(jitter))

        # wrap model parameters as nn.Parameter:
        self.mixes = nn.Parameter(_mixes, requires_grad=True)
        self.means = nn.Parameter(_means, requires_grad=True)
        self.stdvs = nn.Parameter(_stdvs, requires_grad=True)

    def gmm(self):
        """
        Return well-formed parameters; this method is intended to return normalized values
        that represent a proper gaussian mixture model.

        Returned values have `requires_grad == True`.
        """
        pi = F.softmax(self.mixes, dim=0)
        mu = self.means
        sigma = torch.exp(self.stdvs)
        return (pi, mu, sigma)

    def sample(self, size=1):
        """
        Return a random sample from the underlying gaussian mixture model.
        
        Returned values have `requires_grad == False`.
        """
        pi, mu, sigma = self.gmm()
        choices = torch.multinomial(pi, size, replacement=True)
        samples = torch.normal(mu[choices], sigma[choices])
        return (choices, samples)

    def activations(self, x):
        """
        Return a vector giving the probabilities of the sample `x` from each gaussian component.
        Can accept `x` as either a scalar or a vector of batched scalar samples.
        Returned values have `requires_grad == True`.
        """
        return gaussian_distribution(x, self.means, torch.exp(self.stdvs))

    def forward(self, x):
        """
        Compute the probability of `x` in the mixture defined by the underlying parameters.
        """
        pi, mu, sigma = self.gmm()
        return torch.sum(gaussian_distribution(x, mu, sigma) * pi, dim=1)


# ==== helper functions:
oneDivSqrtTwoPI = 1.0 / math.sqrt(2.0*math.pi) # normalisation factor for gaussian.
def gaussian_distribution(ys, mu, sigma):
    """
    Compute gaussian density of vector ys ~ (batch_size,) against mu ~ (num_components,), sigma ~ (num_components,)
    to get per-component probabilities ~ (batch_size, num_components).
    """
    # broadcast subtraction with mean and normalization to sigma
    y_exp = ys.unsqueeze(1).expand(ys.size(0),mu.size(0))
    mu_exp = mu.unsqueeze(0).expand(ys.size(0),mu.size(0))
    sigma_exp = sigma.unsqueeze(0).expand(ys.size(0),mu.size(0))
    result = (y_exp - mu_exp) * torch.reciprocal(sigma_exp)
    result = - 0.5 * (result * result)
    return (torch.exp(result) * torch.reciprocal(sigma_exp)) * oneDivSqrtTwoPI


def mdn_loss(model_proba):
    epsilon = 1e-8
    return torch.mean(-1 * torch.log(epsilon + model_proba))


def regularized_mdn_loss(mdn, sample):
    epsilon = 1e-3
    proba = mdn(sample)
    raw_loss = torch.mean(-1 * torch.log(epsilon + torch.sum(proba, dim=1)))
    l2_penalty = torch.norm(mdn.mixes,p=2) + torch.norm(mdn.means,p=2) + torch.norm(mdn.stdvs,p=2)
    return (raw_loss + l2_penalty)


def full_mdn_loss(out_pi, out_sigma, out_mu, y):
    epsilon = 1e-3
    result = gaussian_distribution(y, out_mu, out_sigma) * out_pi
    result = torch.sum(result, dim=1)
    result = - torch.log(epsilon + result)
    return torch.mean(result)
