"""
A quick demo showing how MixDenoise works: we'll construct an
artificial GMM with 5 components and construct a MixDenoise
layer with 10 components.

We train on samples from the GMM and pass the resulting outputs
to some feedforward networks for dimensionality reduction from 10=>5,
and try to predict which gaussian it came from.
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from mixdenoise import MixDenoise

import numpy as np
from tqdm import tqdm

### GMM sampling:
gmm_mixes = torch.Tensor([0.05, 0.2, 0.3, 0.4, 0.05])
gmm_means = torch.Tensor([ 1.0, 2.0, 3.0, 4.0, 5.0 ])
gmm_stdvs = torch.Tensor([ 0.1, 0.1, 0.1, 0.1, 0.1 ])
def sample_from_gmm(size=1):
    """
    Return a batch of samples from above gaussian components.
    Output shape is (batch_size, 1).
    """
    # draw an integer vector of gaussian component choices:
    choices = torch.multinomial(gmm_mixes, size, replacement=True)
    # get corresponding gaussian samples:
    samples = torch.normal(gmm_means[choices], gmm_stdvs[choices])
    return (choices, samples.unsqueeze(1))


### Build model, loss, optimizer:
mdl = nn.Sequential(MixDenoise(10),
                    nn.Linear(10,25),
                    nn.Sigmoid(),
                    nn.Linear(25,10),
                    nn.Sigmoid(),
                    nn.Linear(10,5))
loss_fn = nn.CrossEntropyLoss()
opt = optim.SGD(mdl.parameters(), lr=0.01, momentum=0.9, nesterov=True)

### Training:
def train():
    NUM_ITERS = 1000000
    PRINT_EVERY = 10000
    BSZ = 32
    for k in tqdm(range(NUM_ITERS)):
        # clear gradients:
        opt.zero_grad()
        # generate a sample:
        comps, samps = sample_from_gmm(size=BSZ)
        x = Variable(samps)
        t = Variable(comps)
        # generate a prediction:
        p = mdl(x)
        # compute loss:
        loss = loss_fn(p,t)
        # backwards:
        opt.step()
        # print:
        if (k % PRINT_EVERY == 0): tqdm.write("{0} | {1}".format(k,loss.data[0]))

    # print weights:
    print("*** Mixture weights:")
    print(gmm_mixes)
    print("*** Learned weights:")
    print([p for p in mdl[1].parameters()])
    print("*** Learned means:")
    print(mdl[0].means.data)
    print("*** Learned Stdvs:")
    print(mdl[0].stdvs.data)

if __name__ == '__main__':
    train()
