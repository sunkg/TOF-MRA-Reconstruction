# -*- coding: utf-8 -*-
"""
@author: sunkg
"""
import torch
import torch.nn as nn
import CNN
import utils
import sensitivity_model
import opt


class IRB(nn.Module):
    def __init__(self, coils_all, img_size, chans=16, stages=3):
        super().__init__()
        
        #### same stepsize for re and im ###
        self.stepsize = nn.Parameter(0.1*torch.rand(1))        
        self.lambdaK = nn.Parameter(0.1*torch.rand(1))
        self.lambdaI = nn.Parameter(0.1*torch.rand(1))
        self.scale = 0.1
        self.LeakyReLU = nn.LeakyReLU()
        self.img_size = img_size
        self.coils_all = coils_all
        self.KCNN = CNN.Unet(in_chans = 2, out_chans = 2, chans = chans, stages = stages)  # CNN for kspace
        self.ICNN = CNN.Unet_Evid(in_chans = 2, chans = chans, stages = stages) # CNN for image space
        self.ConvBlockSM = CNN.ConvBlockSM(in_chans = 2, conv_num = 2) # CNN for sensitivity map update

               
    #### SMRB: SM map refinement block ####
    def SMRB(self, Target_img_f, sens_maps_updated, Target_Kspace_u, mask):
        Target_Kspace_f = utils.sens_expand(Target_img_f, sens_maps_updated)  

        B, C, H, W = sens_maps_updated.shape
        sens_maps_updated_ = sens_maps_updated.reshape(B*C, 1, H, W)
        sens_maps_updated_ = utils.complex_to_chan_dim(sens_maps_updated_)    
        sens_maps_updated_ = self.ConvBlockSM(sens_maps_updated_)
        sens_maps_updated_ = utils.chan_dim_to_complex(sens_maps_updated_) 
        sens_maps_updated_ = sens_maps_updated_.reshape(B, C, H, W)
        sens_maps_updated = sens_maps_updated - self.stepsize*(2*utils.ifftn(mask*(mask*Target_Kspace_f - Target_Kspace_u) * Target_img_f.conj()) + self.scale*sens_maps_updated_)         
        sens_maps_updated = sens_maps_updated / (utils.rss(sens_maps_updated) + 1e-12)
        sens_maps_updated = sens_maps_updated 
        
        return sens_maps_updated   
    
    
    def DC(self, Target_img_f, Target_Kspace_u, sens_maps_updated, mask, lambdaI, lambdaK, ZI, ZK):
        
        lhs = lambda x: utils.sens_reduce(mask*mask*utils.sens_expand(x, sens_maps_updated), sens_maps_updated) + lambdaI.abs()*x + lambdaK.abs()*x
        rhs = utils.sens_reduce(mask*Target_Kspace_u, sens_maps_updated) + lambdaI.abs()*ZI + lambdaK.abs()*utils.ifftn(ZK)

        Target_img_f, iter_num = opt.zconjgrad(Target_img_f, rhs, lhs, max_iter=5)
        
        return Target_img_f
    
    
    def forward(self, Target_Kspace_u, Target_img_f, mask, sens_maps_updated): 
        
        Target_Kspace_f = utils.fftn(Target_img_f)
               
        Target_Kspace_f = utils.complex_to_chan_dim(Target_Kspace_f)
        Target_img_f = utils.complex_to_chan_dim(Target_img_f)

        output_CNN = self.KCNN(Target_Kspace_f)
        gamma, v, alpha = self.ICNN(Target_img_f)
        output_CNN = utils.chan_dim_to_complex(output_CNN) # UNet not NormUnet
     
        #### denormalize and turn back to complex values #### 
        Target_img_f = utils.chan_dim_to_complex(Target_img_f)   
        
        #### run CG for DC ####
        Target_img_f = self.DC(Target_img_f, Target_Kspace_u, sens_maps_updated, mask, self.lambdaI.abs(), self.lambdaK.abs(), gamma, output_CNN)

        sens_maps_updated = self.SMRB(Target_img_f, sens_maps_updated, Target_Kspace_u, mask)
 
        return Target_img_f, v, alpha, sens_maps_updated
    
    
class UnrollUncertain(nn.Module):
    def __init__(self, coils, img_size, chans=16, stages=3,
                 num_recurrent=5, sens_chans=8, sens_steps=4):
        super().__init__()
           
        self.sens_net = sensitivity_model.SMEB(
                        chans=sens_chans,
                        sens_steps=sens_steps
                        )

        self.coils = coils # coils of single modality
        self.num_recurrent = num_recurrent
        self.recurrent = nn.ModuleList([IRB(coils_all=coils, img_size=img_size, chans=chans, stages=stages) for i in range(num_recurrent)])        
            
    def forward(self, Target_Kspace_u, mask, num_low_frequencies):
        recs, vs, alphas = [], [], []
        SMs = []
        if self.coils == 1:
            sens_maps_updated = torch.ones_like(Target_Kspace_u)

        else:
            sens_maps_updated = self.sens_net(Target_Kspace_u, num_low_frequencies)

        Target_img_f = utils.sens_reduce(Target_Kspace_u, sens_maps_updated) # initialization of Target image
        SMs.append(sens_maps_updated)
        recs.append(Target_img_f)
        
        #### DCRB blocks #### 
        for idx, IRB_ in enumerate(self.recurrent):

            #### Update of MR image by IRB ####
            Target_img_f, v, alpha, sens_maps_updated = IRB_(Target_Kspace_u, Target_img_f, mask, sens_maps_updated)
            
            recs.append(Target_img_f)
            vs.append(v)
            alphas.append(alpha)

        return recs, vs, alphas, utils.rss(Target_img_f), sens_maps_updated, Target_img_f
    
    
