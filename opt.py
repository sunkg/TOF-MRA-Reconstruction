#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 17:28:40 2023

@author: sunkg
"""

import torch

# Complex dot product of two complex-valued multidimensional Tensors
def zdot_batch(x1, x2):
    batch = x1.shape[0]
    return torch.reshape(torch.conj(x1)*x2, (batch, -1)).sum(1)

# Same, applied to self --> squared L2-norm
def zdot_single_batch(x):
    return zdot_batch(x, x)

def itemize(x):
    """Converts a Tensor into a list of Python numbers.
    """
    if len(x.shape) < 1:
        x = x[None]
    if x.shape[0] > 1:
        return [xx.item() for xx in x]
    else:
        return x.item()

def zconjgrad(x, b, Aop_fun, max_iter=10, l2lam=0., eps=1e-6, verbose=False):
    """Conjugate Gradient Algorithm for a complex vector space applied to batches; assumes the first index is batch size.
    Args:
    x (complex-valued Tensor): The initial input to the algorithm.
    b (complex-valued Tensor): The residual vector
    Aop_fun (func): A function performing the normal equations, A.H * A
    max_iter (int): Maximum number of times to run conjugate gradient descent.
    l2lam (float): The L2 lambda, or regularization parameter (must be positive).
    eps (float): Determines how small the residuals must be before termination…
    verbose (bool): If true, prints extra information to the console.
    Returns:
    	A tuple containing the output vector x and the number of iterations performed.
    """

    # the first calc of the residual may not be necessary in some cases...
    r = b - (Aop_fun(x) + l2lam * x)
    p = r

    rsnot = zdot_single_batch(r).real
    rsold = rsnot
    rsnew = rsnot

    eps_squared = eps ** 2

    reshape = (-1,) + (1,) * (len(x.shape) - 1)

    num_iter = 0

    for i in range(max_iter):

        if verbose:
            print('{i}: {rsnew}'.format(i=i, rsnew=itemize(torch.sqrt(rsnew))))

        if rsnew.max() < eps_squared:
            break

        Ap   = Aop_fun(p) + l2lam * p
        
        pAp  = zdot_batch(p, Ap).real
        alpha = (rsold / pAp).reshape(reshape)

        x = x + alpha * p
        r = r - alpha * Ap

        rsnew = zdot_single_batch(r).real

        beta = (rsnew / rsold).reshape(reshape)

        rsold = rsnew
    
        p = beta * p + r
        num_iter += 1

    if verbose:
        print('FINAL: {rsnew}'.format(rsnew=torch.sqrt(rsnew)))

    return x, num_iter
