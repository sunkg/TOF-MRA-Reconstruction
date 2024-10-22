# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 09:20:31 2022
@author: sun kaicong 
"""
import torch.nn as nn
from timm.models.layers import to_2tuple, to_3tuple
import math
import torch
import numpy as np
import torch.fft
import torch.nn.functional as F
import nibabel as nib
import h5py
import numbers


def fftn(x):
    if len(x.shape) == 4:
        x = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(x, dim = tuple([-2,-1])), norm='ortho'), dim = tuple([-2,-1]))
    elif len(x.shape) == 5:
        x = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(x, dim = tuple([-3,-2,-1])), norm='ortho'), dim = tuple([-3,-2,-1]))
    return x

def ifftn(x):
    if len(x.shape) == 4:
        x = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(x, dim = tuple([-2,-1])), norm='ortho'), dim = tuple([-2,-1]))
    elif len(x.shape) == 5:
        x = torch.fft.fftshift(torch.fft.ifftn(torch.fft.fftshift(x, dim = tuple([-3,-2,-1])), norm='ortho'), dim = tuple([-3,-2,-1]))
    return x

def rss(x):
    return torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)

def rss2d(x):
    assert len(x.shape) == 2
    return (x.real**2 + x.imag**2).sqrt()
    
def ssimloss(X, Y):
    assert not torch.is_complex(X)
    assert not torch.is_complex(Y)
    win_size = 7
    k1 = 0.01
    k2 = 0.03
    w = torch.ones(1, 1, win_size, win_size).to(X) / win_size ** 2
    NP = win_size ** 2
    cov_norm = NP / (NP - 1)
    data_range = 1
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2
    ux = F.conv2d(X, w)
    uy = F.conv2d(Y, w)
    uxx = F.conv2d(X * X, w)
    uyy = F.conv2d(Y * Y, w)
    uxy = F.conv2d(X * Y, w)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    A1, A2, B1, B2 = (
        2 * ux * uy + C1,
        2 * vxy + C2,
        ux ** 2 + uy ** 2 + C1,
        vx + vy + C2,
    )
    D = B1 * B2
    S = (A1 * A2) / D
    return 1 - S.mean()


def gaussian_kernel_1d(sigma):
    kernel_size = int(2*math.ceil(sigma*2) + 1)
    x = torch.linspace(-(kernel_size-1)//2, (kernel_size-1)//2, kernel_size).cuda()
    kernel = 1.0/(sigma*math.sqrt(2*math.pi))*torch.exp(-(x**2)/(2*sigma**2))
    kernel = kernel/torch.sum(kernel)
    return kernel

def gaussian_kernel_2d(sigma):
    y_1 = gaussian_kernel_1d(sigma[0])
    y_2 = gaussian_kernel_1d(sigma[1])
    kernel = torch.tensordot(y_1, y_2, 0)
    kernel = kernel / torch.sum(kernel)
    return kernel

def gaussian_smooth(img, sigma):
    sigma = max(sigma, 1e-12)
    kernel = gaussian_kernel_2d((sigma, sigma))[None, None, :, :].to(img)
    padding = kernel.shape[-1]//2
    img = torch.nn.functional.conv2d(img, kernel, padding=padding)
    return img

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=3):
        super(GaussianSmoothing, self).__init__()
        sigma = max(sigma, 1e-12)
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        
        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


def pad_imgs(data, pads):
    data = F.pad(data, pad=pads, mode='constant', value=0)
    return data

    
def AddBlur(img, kernelsize = None, sigma = 1e-12, dim = 2):
    
    if sigma <= 1:
        kernelsize = kernelsize or 3
        pad = 1
    elif sigma <= 2:
        kernelsize = kernelsize or 5
        pad = 2
    elif sigma > 2:
        kernelsize = kernelsize or 7
        pad = 3
    else:
        raise Exception("Not implemented!!!")
        
    img = pad_imgs(img, tuple([pad, pad]*dim))

    smoothing = GaussianSmoothing(img.shape[1]*2, kernelsize, sigma, dim) # coil*2 for complex value
        
    img = complex_to_chan_dim(img)
    img = smoothing(img)
    img = chan_dim_to_complex(img)
    return img


def AddNoise(img, noise_mean = 0, noise_std = None): # for complex value
    if noise_std != None:
        img.real = img.real + torch.randn(img.shape) * noise_std[0]
        img.imag = img.imag + torch.randn(img.shape) * noise_std[1]

    return img


def compute_marginal_entropy(values, bins, sigma):
    normalizer_1d = np.sqrt(2.0 * np.pi) * sigma
    sigma = 2*sigma**2
    p = torch.exp(-((values - bins).pow(2).div(sigma))).div(normalizer_1d)
    p_n = p.mean(dim=1)
    p_n = p_n/(torch.sum(p_n) + 1e-10)
    return -(p_n * torch.log(p_n + 1e-10)).sum(), p


def _mi_loss(I, J, bins, sigma):
    # compute marjinal entropy
    ent_I, p_I = compute_marginal_entropy(I.view(-1), bins, sigma)
    ent_J, p_J = compute_marginal_entropy(J.view(-1), bins, sigma)
    # compute joint entropy
    normalizer_2d = 2.0 * np.pi*sigma**2
    p_joint = torch.mm(p_I, p_J.transpose(0, 1)).div(normalizer_2d)
    p_joint = p_joint / (torch.sum(p_joint) + 1e-10)
    ent_joint = -(p_joint * torch.log(p_joint + 1e-10)).sum()

    return -(ent_I + ent_J - ent_joint)


def mi_loss(I, J, bins=64 ,sigma=1.0/64, minVal=0, maxVal=1):
    bins = torch.linspace(minVal, maxVal, bins).to(I).unsqueeze(1)
    neg_mi =[_mi_loss(I, J, bins, sigma) for I, J in zip(I, J)]
    return sum(neg_mi)/len(neg_mi)


def ms_mi_loss(I, J, bins=64, sigma=1.0/64, ms=3, smooth=3, minVal=0, maxVal=1):
    smooth_fn = lambda x: torch.nn.functional.avg_pool2d( \
            gaussian_smooth(x, smooth), kernel_size = 2, stride=2)
    loss = mi_loss(I, J, bins=bins, sigma=sigma, minVal=minVal, maxVal=maxVal)
    for _ in range(ms - 1):
        I, J = map(smooth_fn, (I, J))
        loss = loss + mi_loss(I, J, \
                bins=bins, sigma=sigma, minVal=minVal, maxVal=maxVal)
    return loss / ms


def lncc_loss(I, J, win=None):
    """
    calculate the normalize cross correlation between I and J
    assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    """

    ndims = len(list(I.size())) - 2
    assert ndims ==  2, "volumes should be 2 dimensions. found: %d" % ndims

    if win is None:
        win = [9] * ndims

    sum_filt = torch.ones([1, 1, *win]).to(I)

    pad_no = math.floor(win[0]/2)

    stride = (1,1)
    padding = (pad_no, pad_no)
    
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)

    cc = cross*cross / (I_var*J_var + 1e-5)

    return -1 * torch.mean(cc)


def compute_local_sums(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross


def ms_lncc_loss(I, J, win=None, ms=3, sigma=3):
    smooth_fn = lambda x: torch.nn.functional.avg_pool2d( \
            gaussian_smooth(x, sigma), kernel_size = 2, stride=2)
    loss = lncc_loss(I, J, win)
    for _ in range(ms - 1):
        I, J = map(smooth_fn, (I, J))
        loss = loss + lncc_loss(I, J, win)
    return loss / ms


def correlation_loss(SM):
    B, C, H, W = SM.shape
    SM_ = SM.view(B, C, -1)
    loss = 0
    for i in range(B):
        cc = torch.corrcoef(SM_[i, ...])
        loss += F.l1_loss(cc, torch.eye(C).cuda())
    return loss


def gradient(x,h_x=None,w_x=None):
    if h_x is None and w_x is None:
        h_x = x.size()[2]
        w_x = x.size()[3]
    r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
    l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
    t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
    b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
    xgrad = torch.pow(torch.pow((r - l), 2) + torch.pow((t - b), 2), 0.5)
    return xgrad


def convert(nii_path, h5_path, protocal):
    # convert nii file with path nii_path to h5 file stored at h5_path
    # protocal name as string
    h5 = h5py.File(h5_path, 'w')
    nii = nib.load(nii_path)
    array = nib.as_closest_canonical(nii).get_fdata() #convert to RAS
    array = array.T.astype(np.float32)
    h5.create_dataset('image', data=array)
    h5.attrs['max'] = array.max()
    h5.attrs['acquisition'] = protocal
    h5.close()
    

def rigid_grid(img):
    # rotate batch
    rotation = 2*np.pi*0.5
    affines = []
    r_x = np.random.uniform(-rotation, rotation, img.shape[0])
    r_y = np.random.uniform(-rotation, rotation, img.shape[0])
    r_z = np.random.uniform(-rotation, rotation, img.shape[0])

    for alpha, beta, gamma in zip(r_x, r_y, r_z):
        # convert origin to center
        # rotation
        R = np.array([[np.cos(beta)*np.cos(gamma), np.sin(alpha)*np.sin(beta)*np.cos(gamma)-np.cos(alpha)*np.sin(gamma), np.cos(alpha)*np.sin(beta)*np.cos(gamma)+np.sin(alpha)*np.sin(gamma), 0], 
                [np.cos(beta)*np.sin(gamma), np.sin(alpha)*np.sin(beta)*np.sin(gamma)+np.cos(alpha)*np.cos(gamma), np.cos(alpha)*np.sin(beta)*np.sin(gamma)-np.sin(alpha)*np.cos(gamma), 0], 
                [-np.sin(beta), np.sin(alpha)*np.cos(beta), np.cos(alpha)*np.cos(beta), 0],    
                [0, 0, 0, 1]])

        M = R # the center is already (0,0), no need to T1, T2
        affines.append(M[:-1])
    M = np.stack(affines, 0)
    M = torch.as_tensor(M, dtype=img.dtype).to(img, non_blocking=True)
    grid = torch.nn.functional.affine_grid(M, \
            size=img.shape, align_corners=False)
    return grid


def augment(img, rigid=True, bspline=True, grid=None):
    if grid is None:
        assert rigid == True
        img_abs = img.abs()
        grid = rigid_grid(img_abs)
    else:
        assert rigid == False
        assert bspline == False
    sample = lambda x: torch.nn.functional.grid_sample(x, grid, \
            padding_mode='reflection', align_corners=False, mode='bilinear')
    if torch.is_complex(img):
        img = sample(img.real) + sample(img.imag)*1j
    else:
        img = sample(img)

    return img


def createMIP3D(img, slices_num = 15, axis = -3):
    ''' create the mip image from original image, slice_num is the number of 
    slices for maximum intensity projection'''
    #img_rot = augment(img)
    img_rot = img
    img_shape = img_rot.shape
    mip = torch.zeros(img_shape)
    for batch_idx in range(len(img)):
        for i in range(img_shape[axis]):
            start = max(0, i-slices_num)
            if axis == -3:
                mip[batch_idx, 0, i, :, :] = torch.amax(img_rot[batch_idx, 0, start:i+1], axis)
            elif axis == -2:
                mip[batch_idx, 0, :, i, :] = torch.amax(img_rot[batch_idx, 0, :, start:i+1, :], axis)
            elif axis == -1:
                mip[batch_idx, 0, :, :, i] = torch.amax(img_rot[batch_idx, 0, :, :, start:i+1], axis)
    return mip


def createMIP2D(img, slices_num = 15, axis = -2):
    ''' create the mip image from original image, slice_num is the number of 
    slices for maximum intensity projection'''
    img_shape = img.shape
    mip = torch.zeros_like(img)
    for batch_idx in range(len(img)):
        for i in range(img_shape[axis]):
            start = max(0, i-slices_num)
            if axis == -2:
                mip[batch_idx, 0, i, :] = torch.amax(img[batch_idx, 0, start:i+1], axis)
            elif axis == -1:
                mip[batch_idx, 0, :, i] = torch.amax(img[batch_idx, 0, :,start:i+1], axis)
    return mip


def complex_to_chan_dim(x: torch.Tensor) -> torch.Tensor:
    assert torch.is_complex(x)
    return torch.cat([x.real, x.imag], dim=1)


def chan_dim_to_complex(x: torch.Tensor) -> torch.Tensor:
    assert not torch.is_complex(x)
    if len(x.shape) == 4:
        _, c, _, _ = x.shape
    elif len(x.shape) == 5:
        _, c, _, _, _ = x.shape
    assert c % 2 == 0
    c = c // 2
    return torch.complex(x[:,:c], x[:,c:])


def UpImgComplex(img_complex, SR_scale):
    img_real=complex_to_chan_dim(img_complex)
    img_real=nn.functional.interpolate(img_real, scale_factor=SR_scale, mode='bicubic')
    return chan_dim_to_complex(img_real)


def norm(x: torch.Tensor):
    # group norm
    b, c, h, w = x.shape
    assert c%2 == 0
    x = x.view(b, 2, c // 2 * h * w)

    mean = x.mean(dim=2).view(b, 2, 1)
    std = x.std(dim=2).view(b, 2, 1)

    x = (x - mean) / (std + 1e-12)

    return x.view(b, c, h, w), mean, std
    

def unnorm(
    x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) :
    b, c, h, w = x.shape
    assert c%2 == 0
    x = x.view(b, 2, c // 2 * h * w)
    x = x* std + mean
    return x.view(b, c, h, w)
        

def preprocess(x):
    assert torch.is_complex(x)
    #x, mean, std = norm(x)
    x = complex_to_chan_dim(x)
    x, mean, std = norm(x)
    return x, mean, std


def postprocess(x, mean, std):
    x = unnorm(x, mean, std)
    x = chan_dim_to_complex(x)
    return x
  
    
def pad(x, window_size):
    if len(x.shape) == 4:
        _, _, h, w = x.size()
        mod_pad_h = (window_size - h % window_size) % window_size
        mod_pad_w = (window_size - w % window_size) % window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x, (mod_pad_w, mod_pad_h)
    elif len(x.shape) == 5:
        _, _, d, h, w = x.size()
        mod_pad_d = (window_size - d % window_size) % window_size
        mod_pad_h = (window_size - h % window_size) % window_size
        mod_pad_w = (window_size - w % window_size) % window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h, 0, mod_pad_d), 'reflect')
        return x, (mod_pad_w, mod_pad_h, mod_pad_d)


def unpad(
    x: torch.Tensor,
    w_pad: int,
    h_pad: int
) -> torch.Tensor:
    return x[...,0 : x.shape[-2] - h_pad, 0 : x.shape[-1] - w_pad]
 

def check_image_size(img_size, window_size):
    h, w = img_size
    mod_pad_h = (window_size - h % window_size) % window_size
    mod_pad_w = (window_size - w % window_size) % window_size
    return h + mod_pad_h, w + mod_pad_w
    

def sens_expand(image: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    return fftn(image * sens_maps)


def sens_reduce(kspace: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    return (ifftn(kspace) * sens_maps.conj()).sum(dim=1, keepdim=True)
    
    
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out
    

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

								
								
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
