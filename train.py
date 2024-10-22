#!/usr/bin/env python-3
"""
@author: sunkg
"""
import os, random
import time
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import numpy as np
import tqdm
from paired_dataset import get_train_volume_datasets, get_eval_volume_datasets
from basemodel import Config
from model import ReconModel
import utils

    
def main(args):
    # setup
    cfg = Config()
    cfg.sparsity = args.sparsity
    cfg.lr = args.lr
    cfg.img_size = tuple(args.shape)
    cfg.coils = args.coils
    cfg.mask = args.mask
    cfg.batch_size = args.batch_size
    cfg.start = args.start
    cfg.CL_num = tuple(args.CL_num)
    cfg.use_amp = args.use_amp
    cfg.num_recurrent = args.num_recurrent
    cfg.chans = args.chans 
    cfg.stages = args.stages
    cfg.lambda0 = args.lambda0
    cfg.lambda1 = args.lambda1
    cfg.lambda2 = args.lambda2
    cfg.lambda3 = args.lambda3
    cfg.lambda4 = args.lambda4
    cfg.GT = args.GT
    cfg.unrolling = args.unrolling
    cfg.sens_chans = args.sens_chans 
    cfg.sens_steps = args.sens_steps
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'     
  
    print(cfg)

    for path in [args.logdir, args.logdir+'/res', args.logdir+'/ckpt']:
        if not os.path.exists(path):
            os.mkdir(path)
            print('mkdir:', path)

    print('Loading model...')
    os.environ["CUBLAS_WORKSPACE_CONFIG"]= ':4096:8'
    #torch.use_deterministic_algorithms(True)
    seed = 14982321 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.device = device.type
    if cfg.device == 'cpu': cfg.GPUs = 0 
    else: cfg.GPUs = 1
    
    batchsize_train = args.batch_size
    iter_cnt = 0

    net = ReconModel(cfg=cfg)
    epoch_start = 0

    net = net.to(device)

    print('Loading data...')
    volumes_val = get_eval_volume_datasets(basepath = args.evalpath, crop=cfg.img_size, mask = cfg.mask)
    slices_val = torch.utils.data.ConcatDataset(volumes_val)
    print('Finished loading.')

    print('training...')
    last_loss, last_ckpt = 0, 0
    time_data, time_vis = 0, 0
    signal_earlystop = False
    iter_best = iter_cnt
    Eval_PSNR_best = None
    Eval_SSIM_best = None

    optim_R = torch.optim.AdamW(net.parameters(), \
            lr=cfg.lr, weight_decay=0)
    
    scalar = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
    
    Eval_PSNR, Eval_SSIM = [], []
    time_start = time.time()
    
    for index_epoch in tqdm.trange(epoch_start, args.epoch, desc='epoch', leave=True):
        volumes_train = []
        volumes_train = get_train_volume_datasets(basepath = args.trainpath, crop=cfg.img_size, mask=cfg.mask)
        slices_train = torch.utils.data.ConcatDataset(volumes_train)
        

        if index_epoch == 0:
            print('dataset: ' \
                + str(len(slices_train)) + ' / ' \
                + str(len(volumes_train)) + ' for training, ' \
                + str(len(slices_val)) + ' / ' \
                + str(len(volumes_val)) + ' for validation')

        loader_train = torch.utils.data.DataLoader( \
            slices_train, batch_size=batchsize_train, shuffle=True, \
            num_workers=args.num_workers, pin_memory=False, drop_last=True)
        loader_val = torch.utils.data.DataLoader( \
            slices_val, batch_size=1, shuffle=True, \
            num_workers=args.num_workers, pin_memory=False, drop_last=False)

        ###################  training ########################
        tqdm_iter = tqdm.tqdm(loader_train, desc='iter', \
                bar_format=str(batchsize_train)+': {n_fmt}/{total_fmt}'+\
                '[{elapsed}<{remaining},{rate_fmt}]'+'{postfix}', leave=False)
        
        #### learning rate decay ####
        if index_epoch%50 == 0:
            for param_group in optim_R.param_groups:
                param_group['lr'] = param_group['lr']*(0.5**(index_epoch//50))
                
        if signal_earlystop:
            break

        net.train()

        for batch in tqdm_iter:

            time_data = time.time() - time_start

            iter_cnt += 1
            with torch.no_grad():
                batch = [x.to(device, non_blocking=True) for x in batch]    
                batch = [utils.complex_to_chan_dim(x) for x in batch]            
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                loss_all = net(*batch, train_flag = True)
            
            optim_R.zero_grad()
            scalar.scale(loss_all.mean()).backward()
            scalar.step(optim_R)
            scalar.update()
            
            del batch
            del loss_all

            time_start = time.time()

            time_vis = time.time() - time_start
            time_start = time.time()
            postfix = '[%d/%d/%d]'%( \
                    iter_cnt, last_loss, last_ckpt)
            if time_data >= 0.1:
                postfix += ' data %.1f'%time_data
            if time_vis >= 0.1:
                postfix += ' vis %.1f'%time_vis
            tqdm_iter.set_postfix_str(postfix)


        ###################  validation  ########################
        net.eval()

        tqdm_iter = tqdm.tqdm(loader_val, desc='iter', \
                bar_format=str(args.batch_size)+'(val) {n_fmt}/{total_fmt}'+\
                '[{elapsed}<{remaining},{rate_fmt}]'+'{postfix}', leave=False)
        stat_loss = []
        time_start = time.time()
        with torch.no_grad():
            for batch in tqdm_iter:
                time_data = time.time() - time_start
                batch = [x.to(device, non_blocking=True) for x in batch]
                batch = [utils.complex_to_chan_dim(x) for x in batch]   
                net.test(*batch)
                stat_loss.append(net.Eval)

                del batch

                time_start = time.time()
                if time_data >= 0.1:
                    postfix += ' data %.1f'%time_data

            Eval_PSNR_current, Eval_SSIM_current = [(sum(i)/len(loader_val)) for i in zip(*stat_loss)]
            Eval_PSNR.append(Eval_PSNR_current)
            Eval_SSIM.append(Eval_SSIM_current)
        
            np.save(args.logdir+'/PSNR', np.array(Eval_PSNR))
            np.save(args.logdir+'/SSIM', np.array(Eval_SSIM))      
            print('Current loss is %.4f, loss_fidelity is %.4f, loss_evid %.4f, PSNR %.2f, SSIM %.4f'%(net.loss_all.detach().item(), net.loss_fidelity.detach().item(), net.loss_evid.detach().item(), Eval_PSNR_current, Eval_SSIM_current))

            if (Eval_PSNR_best is None) or ((Eval_PSNR_current > Eval_PSNR_best) & (Eval_SSIM_current > Eval_SSIM_best)):
                Eval_PSNR_best = Eval_PSNR_current
                Eval_SSIM_best = Eval_SSIM_current
                epoch_best = index_epoch
                print('Current best epoch %d/%d:'%(epoch_best, args.epoch), f' PSNR: {Eval_PSNR_best:.2f}', f', SSIM: {Eval_SSIM_best:.4f}')
                torch.save({'state_dict': net.state_dict(),
                            'config': cfg,
                            'epoch': index_epoch},
                             args.logdir+'/ckpt/best.pt')  #save best model variant

            else:
                if iter_cnt >= args.early_stop + iter_best:
                    signal_earlystop=True
                    print('signal_earlystop set due to early_stop')
                
                      
    print('reached end of training loop, and signal_earlystop is '+str(signal_earlystop))

    


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='TOF-MRA reconstruction')
    parser.add_argument('--logdir', metavar='logdir', \
                        type=str, default='/path_to_save',\
                        help='log directory')
    parser.add_argument('--epoch', type=int, default=100, \
                        help='epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, \
                        help='batch size for training')
    parser.add_argument('--num_workers', type=int, default=0, \
                        help='number of threads for parallel preprocessing')
    parser.add_argument('--lr', type=float, default=1e-4, \
                        help='learning rate')
    parser.add_argument('--early_stop', type=int, default=1000000, metavar='N', \
                        help='stop training after val loss not going down for N iters')
    parser.add_argument('--lambda0', type=float, default=10, \
                        help='weight of the kspace loss')
    parser.add_argument('--lambda1', type=float, default=1, \
                        help='weight of the SSIM loss')
    parser.add_argument('--lambda2', type=float, default=1e2, \
                        help='weight of the TV Loss')
    parser.add_argument('--lambda3', type=float, default=5, \
                        help='weight of the MIP Loss')  
    parser.add_argument('--lambda4', type=float, default=0.02, \
                       help='weight of the Evidence Loss') 
    parser.add_argument('--shape', type=int, nargs='+', \
                        help='image shape')
    parser.add_argument('--CL_num', type=int, nargs='+', \
                        help='2D dimension of ACS lines')
    parser.add_argument('--num_recurrent', type=int, default=9, \
                        help='number of DCRBs')
    parser.add_argument('--sens_chans', type=int, default=8, \
                        help='number of channels in sensitivity network')
    parser.add_argument('--sens_steps', type=int, default=4, \
                        help='number of steps in initial sensitivity network')
    parser.add_argument('--chans', type=int, default=32, \
                        help='number of channels in Unet')
    parser.add_argument('--stages', type=int, default=4, \
                        help='number of stages in Unet')
    parser.add_argument('--GT', type=bool, default=True, \
                        help='if there is GT, default is True') 
    parser.add_argument('--mask', metavar='type', \
                        choices=['mask', 'Random', 'Equispaced', 'VD', 'Loupe'], \
                        type=str, default = 'Equispaced', help='types of mask')
    parser.add_argument('--start', type=int, default=0, \
                        help='index of starting slice')
    parser.add_argument('--sparsity', metavar='0-1', \
                        type=float, default=0.25, help='sparisity of masks for target modality')
    parser.add_argument('--trainpath', default='/path_to_training_data')
    parser.add_argument('--evalpath', default='/path_to_validation_data')
    parser.add_argument('--coils', type=int, default=12, \
                        help='number of coils')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--unrolling', action='store_true')

    args = parser.parse_args()

    main(args)

