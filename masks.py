#!/usr/bin/env python-3
"""
@author: sunkg, updated from the work of XuanKai: https://github.com/woxuankai/SpatialAlignmentNetwork
"""
import math
import functools
import random
import torch
import numpy as np

class Mask(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.register_parameter('weight', None) # weights
        self.weight = torch.nn.Parameter(torch.ones(shape))
        # if a weight is already pruned, set to True
        self.register_buffer('pruned', None)
        self.pruned = torch.zeros(shape, dtype=torch.bool)

    def forward(self, image):
        mask = torch.ones_like(self.weight)
        mask.masked_scatter_(self.pruned, torch.zeros_like(self.weight))
        # unable to set a leaf variable here
        #self.weight.masked_scatter_(self.pruned, torch.zeros_like(self.weight))
        # mask weight in preventation of weight changing
        return image * (self.weight*mask)[None, None, None, :]


class RandomMask(Mask):
    """When  the  acceleration factorequals four,
    the fully-sampled central region includes 8% of all k-space lines;
    when it equals eight, 4% of all k-space lines are included.
    """
    def __init__(self, sparsity, shape, CL_num):
        """
        sparsity: float, desired sparsity, can only be either 1/4 or 1/8
        shape: int, output mask shape
        """
        super().__init__(shape)
        self.pruned = torch.zeros(shape, dtype=torch.bool)
        other_ratio = (sparsity*shape[1] - CL_num[1])/(shape[1] - CL_num[1])
        prob = torch.ones(shape[1])*2
        # low freq is of the border
        prob[shape[1]//2-CL_num[1]//2:shape[1]//2-CL_num[1]//2+CL_num[1]] = other_ratio
        thresh = torch.rand(shape[1])
        _, ind = torch.topk(prob - thresh, math.floor(sparsity*shape[1]-CL_num[1]), dim=-1)
        self.pruned[:,ind] = True
        self.pruned[shape[0]//2-CL_num[0]//2:shape[0]//2-CL_num[0]//2+CL_num[0], shape[1]//2-CL_num[1]//2:shape[1]//2-CL_num[1]//2+CL_num[1]] = True



class EquispacedMask(Mask):
    def __init__(self, sparsity, shape, CL_num):
        """
        Evaluate if the sum equals sparsity for the used case
        """
        super().__init__(shape)
        pruned = np.zeros(shape, dtype=np.bool)
        pruned[shape[0]//2-CL_num[0]//2:shape[0]//2-CL_num[0]//2+CL_num[0], shape[1]//2-CL_num[1]//2:shape[1]//2-CL_num[1]//2+CL_num[1]] = True
        rest_one = int(shape[-1]*sparsity)-CL_num[-1]
        rest_zero = shape[-1]-CL_num[-1]
        space = rest_zero//rest_one
        pruned[shape[0]//2-CL_num[0]//2:shape[0]//2-CL_num[0]//2+CL_num[0],shape[1]//2-CL_num[1]//2-space::-space] = True
        span_right = space * (int(shape[-1]*sparsity) - pruned[0,:].sum())
        pruned[shape[0]//2-CL_num[0]//2:shape[0]//2-CL_num[0]//2+CL_num[0],shape[1]//2+CL_num[1]//2+space:shape[1]//2+CL_num[1]//2+span_right+1:space] = True

        self.pruned = torch.from_numpy(pruned)




class VDMask(Mask):
    def __init__(self, sparsity, shape):
        """
        sparsity: float, desired sparsity, can only be either 1/4 or 1/8
        shape: int, output mask shape
        """
        super().__init__(shape)
        if sparsity == 0.33:     
            pa, pb, AF = 7, 3, 3 #7, 1.8, 3 #3x
        elif sparsity == 0.25:     
            pa, pb, AF = 10, 3, 4 #10, 1.8, 4 #4x
        elif sparsity == 0.125: 
            pa, pb, AF = 16, 3, 8 #16, 1.8, 8 #8x
        else:
            raise ValueError('Not implemented.')

        x_cord, y_cord = torch.meshgrid(torch.arange(-shape[0]//2,shape[0]//2), torch.arange(-shape[1]//2,shape[1]//2))
        mask = torch.exp(-pa*(torch.sqrt(x_cord**2/(shape[0]**2) + y_cord**2/(shape[1]**2)))**pb)
        mask_normalized = mask/(mask.sum()/(shape[0]*shape[1]/AF));
        self.pruned = mask_normalized>torch.rand(shape)


def rescale_prob(x, sparsity):

    xbar = x.mean()
    if xbar > sparsity:
        return x * sparsity / xbar
    else:
        return 1 - (1 - x) * (1 - sparsity) / (1 - xbar)


class LOUPEMask(torch.nn.Module):
    def __init__(self, sparsity, shape, CL_num, pmask_slope=5, sample_slope=48):
        """
        sparsity: float, desired sparsity
        shape: int, output mask shape
        sample_slope: float, slope for soft threshold
        mask_param -> (sigmoid+rescale) -> pmask -> (sample) -> mask
        """
        super().__init__()
        assert sparsity <= 1 and sparsity >= 0
        self.sparsity = sparsity
        self.shape = shape
        self.pmask_slope = pmask_slope
        self.sample_slope = sample_slope
        self.register_parameter('weight', None) # weights
        # eps could be very small, or somethinkg like eps = 1e-6
        # the idea is how far from the tails to have your initialization.
        eps = 1e-6
        #### use random mask as initialization ####
        self.ini_mask = RandomMask(sparsity, shape, CL_num).pruned
        
        x = torch.abs(self.ini_mask[0,:].float()-eps)  #initial mask

        # logit with slope factor
        self.weight = torch.nn.Parameter( \
                -torch.log(1. / x - 1.) / self.pmask_slope)
       

    def forward(self):

        example=torch.ones(1, 1, self.shape[0], self.shape[1]).cuda()
        pmask = rescale_prob( \
                torch.sigmoid(self.weight*self.pmask_slope), \
                self.sparsity)
        thresh = torch.rand(example.shape[0], self.shape[-1]).to(pmask)
        _, ind = torch.topk(pmask - thresh, \
                int(self.sparsity*self.shape[-1]+0.5), dim=-1)
        pruned = torch.zeros_like(thresh).scatter( \
                -1, ind, torch.ones_like(thresh))
       
        if self.training:
            return example*torch.sigmoid((pmask - thresh) * self.sample_slope)[:, None, None, :]
        else:
            return example*(pruned)[:, None, None, :]


if __name__ == '__main__':
    sparsity = 1.0/8
    print(sparsity)
    shape = tuple([320,320])
    CL_num = tuple([shape[0],13]) # 320*0.04=13; 320*0.08 = 26
    random_mask = RandomMask(sparsity, shape, CL_num)
    print(random_mask.pruned.numpy().astype(np.float).mean())
    equispaced = EquispacedMask(sparsity, shape, CL_num)
    print(equispaced.pruned.numpy().astype(np.float).mean())


