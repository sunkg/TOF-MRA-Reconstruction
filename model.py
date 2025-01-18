# -*- coding: utf-8 -*-
"""
@author: sunkg
"""
import torch
import torch.fft
import torch.nn.functional as F
from masks import Mask,  EquispacedMask, LOUPEMask, VDMask, RandomMask
from basemodel import BaseModel
import metrics
import torch.nn as nn
import fD2RT
from utils import rss, fftn, ifftn, ssimloss, createMIP2D
import utils


def gradient_loss(s):
    assert s.shape[-1] == 2, 'not 2D grid?'
    dx = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    dy = torch.abs(s[:, 1:, :, :] - s[:, :-1, :, :])
    dy = dy*dy
    dx = dx*dx
    d = torch.mean(dx)+torch.mean(dy)
    return d/2.0

def generate_rhos(num_recurrent):
    rhos = [0.85**i for i in range(num_recurrent-1,-1,-1)]
    return rhos


def TV_loss(img, weight):
     bs_img, c_img, h_img, w_img = img.size()
     tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
     tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
     return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)


def TV_loss3D(img, weight):
     bs_img, c_img, d_img, h_img, w_img = img.size()
     tv_h = torch.pow(img.mean(dim=2)[:,:,1:,:]-img.mean(dim=2)[:,:,:-1,:], 2).sum()
     tv_w = torch.pow(img.mean(dim=2)[:,:,:,1:]-img.mean(dim=2)[:,:,:,:-1], 2).sum()
     return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)


def NLL_loss(Target_img_f, gamma, v, alpha, lamb=1e1):
    eps = 1e-9
    Omiga = 2*(1+v)
    diff = gamma - Target_img_f
    NLL1 = 0.5*torch.log(torch.pi/(v+eps)) - alpha*torch.log(Omiga+eps)
    NLL2 = (alpha + 0.5)*torch.log((gamma.abs() - Target_img_f.abs())**2*v + Omiga + eps)

    NLL3 = torch.lgamma(alpha+eps)-torch.lgamma(alpha+0.5)
    NLL = NLL1 + NLL2 + NLL3
    KL = (diff.real**2 + diff.imag**2)*(2*v + alpha)

    loss_all = NLL + lamb*KL
    loss_all = loss_all.mean()
    
    return loss_all

                
              
def weighted_mse_loss(source, target, weight):
    return torch.mean(weight * ((source.real - target.real)**2 + (source.imag - target.imag)**2))

  
masks = {"mask": Mask,
        "Random": RandomMask,
        "Equispaced": EquispacedMask,
        "VD":VDMask,
        "Loupe": LOUPEMask
        }

    
class ReconModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rhos = generate_rhos(self.cfg.num_recurrent)
        self.device = self.cfg.device
        if (self.cfg.mask == 'Equispaced') or (self.cfg.mask == 'Loupe') or (self.cfg.mask == 'Random'):
            self.num_low_frequencies = self.cfg.CL_num  # default 
            self.net_mask = masks[self.cfg.mask](self.cfg.sparsity, tuple([self.cfg.img_size[-2], self.cfg.img_size[-1]]), tuple([self.cfg.CL_num[0], self.cfg.CL_num[1]])).to(self.device)
        elif self.cfg.mask == 'VD':
            self.num_low_frequencies = self.cfg.CL_num  # default 
            self.net_mask = masks[self.cfg.mask](self.cfg.sparsity, tuple([self.cfg.img_size[-2], self.cfg.img_size[-1]])).to(self.device)

        self.unrolling = self.cfg.unrolling
  
        self.net_R = fD2RT.UnrollUncertain(coils=self.cfg.coils, img_size=self.cfg.img_size, chans=self.cfg.chans, stages=self.cfg.stages,
                     num_recurrent=self.cfg.num_recurrent, sens_chans=self.cfg.sens_chans, sens_steps=self.cfg.sens_steps).to(self.device) # 2*coils for target and reference
        
        
    def set_input_noGT(self, Subsample_img):

        B, C, H, W = Subsample_img.shape
        self.Target_f_rss = torch.zeros([B, 1, H, W]).to(self.device)
        self.Target_Kspace_f = torch.zeros([B, C, H, W], dtype=torch.complex64).to(self.device)
            
        self.Subsample_Kspace = fftn(Subsample_img)                
                        
        self.Subsample_rss = rss(Subsample_img)
        
        if self.cfg.mask != 'Loupe':
            self.mask = self.net_mask.pruned
        else:
            self.mask = self.Subsample_Kspace != 0
            self.mask = self.mask.to(self.device)


    def set_input_GT(self, Target_img_f, Subsample_img = None):
        
        self.Target_f_rss = rss(Target_img_f)
        self.Target_Kspace_f = fftn(Target_img_f)

        if self.cfg.mask != 'Loupe':
            self.mask = self.net_mask.pruned
        else:
            self.mask = self.net_mask()
            
        if Subsample_img is None:     
            self.Subsample_Kspace = self.Target_Kspace_f * self.mask
            Subsample_img = ifftn(self.Subsample_Kspace) 
            self.Subsample_rss = rss(Subsample_img)
        else:
            self.Subsample_Kspace = fftn(Subsample_img) 
            self.Subsample_rss = rss(Subsample_img)
                 

    def forward(self, Target_img_f = None, Subsample_img = None, train_flag = True):
           
        if train_flag:
            self.net_mask.train() 
        else:
            self.net_mask.eval()

        if Target_img_f != None:
            if not torch.is_complex(Target_img_f):
                Target_img_f = utils.chan_dim_to_complex(Target_img_f)
        if Subsample_img != None:
            if not torch.is_complex(Subsample_img):
                Subsample_img = utils.chan_dim_to_complex(Subsample_img)
       
        with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):        
            if self.cfg.GT == True:
                self.set_input_GT(Target_img_f, Subsample_img)
            else:
                self.set_input_noGT(Subsample_img = Subsample_img)

            self.recs_complex, self.vs, self.alphas, self.rec_rss, self.sens_maps, self.rec_img = self.net_R(\
                Target_Kspace_u = self.Subsample_Kspace,
                mask = self.mask,
                num_low_frequencies = self.num_low_frequencies
                )

            eps = 1e-6
            self.aleatoric_rss = 1/(self.alphas[-1]-1+eps)
            self.epistemic_rss = 1/(self.vs[-1]+eps)/(self.alphas[-1]-1+eps)

            # Record loss
            if self.training:
                self.loss_all = 0
                self.loss_fidelity, self.loss_evid, self.loss_physics = 0, 0, 0
                self.local_fidelities = []
                mse = nn.MSELoss()
                for i in range(self.cfg.num_recurrent):
                    loss_fidelity = F.l1_loss(rss(self.recs_complex[i]),self.Target_f_rss)+self.cfg.lambda0*F.l1_loss(utils.sens_expand(self.recs_complex[i], self.sens_maps), self.Target_Kspace_f)
                    self.local_fidelities.append(self.rhos[i]*loss_fidelity)
                    self.loss_fidelity += self.local_fidelities[-1]
                    Target_img_comb = (Target_img_f * self.sens_maps.conj()).sum(dim=1, keepdim=True)
                    self.loss_evid += self.rhos[i]*NLL_loss(Target_img_comb, self.recs_complex[i], self.vs[i], self.alphas[i])

                self.loss_ssim = self.cfg.lambda1 * ssimloss(self.rec_rss, self.Target_f_rss)
                self.loss_TV = TV_loss(torch.abs(self.sens_maps), self.cfg.lambda2)
                self.mip_rss_axis0 = createMIP2D(self.rec_rss, slices_num = 200, axis=-2)
                self.mip_rss_axis1 = createMIP2D(self.rec_rss, slices_num = 200, axis=-1)
                self.mip_gt_axis0 = createMIP2D(self.Target_f_rss, slices_num = 200, axis=-2)
                self.mip_gt_axis1 = createMIP2D(self.Target_f_rss, slices_num = 200, axis=-1)
                self.loss_mip = self.cfg.lambda3*(mse(self.mip_rss_axis0, self.mip_gt_axis0) + mse(self.mip_rss_axis1, self.mip_gt_axis1))
                self.loss_all += self.loss_mip
                self.loss_all += self.loss_TV
                self.loss_all += self.loss_ssim
                self.loss_evid = self.cfg.lambda4 *self.loss_evid
                self.loss_all += self.loss_evid
     
                return self.loss_all
        

    def test(self, Target_img_f = None, Subsample_img = None):

        assert self.training == False
        self.net_mask.eval()

        with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
            with torch.no_grad():
                self.forward(Target_img_f, Subsample_img, train_flag = False)
                self.metric_PSNR = metrics.psnr(self.Target_f_rss, self.rec_rss, self.cfg.GT)
                self.metric_SSIM = metrics.ssim(self.Target_f_rss, self.rec_rss, self.cfg.GT)
                self.metric_MAE = metrics.mae(self.Target_f_rss, self.rec_rss, self.cfg.GT)
                self.metric_MSE = metrics.mse(self.Target_f_rss, self.rec_rss, self.cfg.GT)
                
                self.metric_PSNR_raw = metrics.psnr(self.Target_f_rss, self.Subsample_rss, self.cfg.GT)
                self.metric_SSIM_raw = metrics.ssim(self.Target_f_rss, self.Subsample_rss, self.cfg.GT)
                self.Eval = tuple([self.metric_PSNR, self.metric_SSIM])

