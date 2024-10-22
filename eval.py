#!/usr/bin/env python3

import os
import torch
import numpy as np
import nibabel as nib
from paired_dataset import get_eval_volume_datasets, center_crop
from model import ReconModel
import random
import metrics


def createMIP2D(img, slices_num = 240, axis = -2):
    #create the mip image from original image, slice_num is the number of slices for maximum intensity projection
    img_shape = img.shape
    mip = torch.zeros_like(img)
    for i in range(img_shape[axis]):
        start = max(0, i-slices_num)
        if axis == -2:
            mip[:, 0, i, :] = torch.amax(img[:, 0, start:i+1, :], axis)
        elif axis == -1:
            mip[:, 0, :, i] = torch.amax(img[:, 0, :, start:i+1], axis)
        elif axis == 0:
            mip[i, 0, :, :] = torch.amax(img[start:i+1, 0, :, :], axis)

    return mip



def main(args):
    
    affine = np.eye(4)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
            
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    os.environ["CUBLAS_WORKSPACE_CONFIG"]= ':4096:8'
    #torch.use_deterministic_algorithms(True)
    seed = 14982321
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if os.path.isfile(args.model_path) or os.path.isdir(args.model_path):
        ckpt = torch.load(args.model_path)
        cfg = ckpt['config']
        net = ReconModel(cfg=cfg)
        net.load_state_dict(ckpt['state_dict'])
        print('load ckpt from:', args.model_path)
    else:
        raise FileNotFoundError
        
    net.use_amp = False
    cfg = net.cfg
    net.GT = args.GT

    volumes = get_eval_volume_datasets(basepath = args.pathtest, crop=cfg.img_size, mask = cfg.mask)
    net.eval()

    PSNRs, SSIMs = [], []
    PSNR_raws, SSIM_raws = [], []
    parts_num = 50
    
    print(args)

    total = sum([param.nelement() for param in net.parameters()])
    print('Network size is %.2fM' % (total/1e6))
    
    for i, volume in enumerate(volumes):
        
        gt, sampled, rec, aleatoric, epistemic = [], [], [], [], []
        
        with torch.no_grad():
        
            batch = [torch.tensor(np.stack(s, axis=0)) for s in \
                     zip(*[volume[j] for j in range(len(volume))])]
    
            batch = [center_crop(i, cfg.img_size) for i in batch]
            
            for part_idx in range(parts_num): #split into 50 times to forward
                vol_size = len(batch[0]) // parts_num
                if part_idx != parts_num -1:
                    vol_part = batch[0][part_idx*vol_size : (part_idx+1)*vol_size]
                else:
                    vol_part = batch[0][part_idx*vol_size:]
                
                batch_ = [vol_part.to(device)]
                net.test(*batch_)
    
                gt.append(net.Target_f_rss.cpu().detach())
                sampled.append(net.Subsample_rss.cpu().detach())
                rec.append(net.rec_rss.cpu().detach())

                aleatoric.append(net.aleatoric_rss.cpu().detach())
                epistemic.append(net.epistemic_rss.cpu().detach())

            del batch            
        
        gt = torch.concat(gt)
        rec = torch.concat(rec)
        sampled = torch.concat(sampled)
        aleatoric = torch.concat(aleatoric)
        epistemic = torch.concat(epistemic)

        PSNR = metrics.psnr(gt, rec, True)
        SSIM = metrics.ssim(gt, rec, True)
        PSNR_raw = metrics.psnr(gt, sampled, True)
        SSIM_raw = metrics.ssim(gt, sampled, True)
        print('Raw volume:',i+1, f', PSNR: {PSNR_raw:.2f}', f', SSIM: {SSIM_raw:.4f}' )
        print('Recon volume:',i+1, f', PSNR: {PSNR:.2f}', f', SSIM: {SSIM:.4f}' )

        PSNRs.append(PSNR)
        SSIMs.append(SSIM)
        PSNR_raws.append(PSNR_raw)
        SSIM_raws.append(SSIM_raw)

        if args.save_img == False:
            continue

        gt, sampled, rec = [nib.Nifti1Image(x.squeeze(1).numpy().T, affine) for x in (gt, sampled, rec)]
        nib.save(gt, args.save_path+'/'+str(i)+'_gt.nii')
        nib.save(sampled, args.save_path+'/'+str(i)+'_sampled.nii')
        nib.save(rec, args.save_path+'/'+str(i)+'_rec.nii')

        aleatoric, epistemic = [nib.Nifti1Image(x.squeeze(1).numpy().T, affine) for x in (aleatoric, epistemic)]
        nib.save(aleatoric, args.save_path+'/'+str(i)+'_aleatoric.nii')
        nib.save(epistemic, args.save_path+'/'+str(i)+'_epistemic.nii')

    print('mean PSNR %.2f(%.2f)/%.2f(%.2f)'%(np.mean(PSNR_raws), np.std(PSNR_raws), np.mean(PSNRs), np.std(PSNRs)))
    print('mean SSIM %.4f(%.4f)/%.4f(%.4f)'%(np.mean(SSIM_raws), np.std(SSIM_raws), np.mean(SSIMs), np.std(SSIMs)))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='TOF-MRA reconstruction')
    parser.add_argument('--model_path', type=str, default=None, \
                        help='with ckpt path, set empty str to load latest ckpt')
    parser.add_argument('--save_path', type=str, default=None, \
                        help='path to save evaluated data')
    parser.add_argument('--save_img', default= False, \
                        type=bool, help='save images or not')
    parser.add_argument('--pathtest', default='/public_bme/data/KaicongSun/Data/301/TOF-MRAraw300um_test', \
                        type=str, help='path to csv of test data')
    parser.add_argument('--GT', type=bool, default=True, \
                        help='if there is GT, default is True') 
    args = parser.parse_args()

    main(args)

