import numpy as np
from matplotlib import pyplot as plt
import argparse, skimage.io
from skimage import feature
from scipy.ndimage import convolve
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def save2img_rgb(img_data, img_fn):
    plt.figure(figsize=(img_data.shape[1]/10., img_data.shape[0]/10.))
    plt.axes([0, 0, 1, 1])
    plt.imshow(img_data, )
    plt.axis('off')
    plt.savefig(img_fn, facecolor='black', edgecolor='black', dpi=10)
    plt.close()

def save2img(d_img, fn):
    if fn[-4:] == 'tiff': 
        img_norm = d_img.copy()
    else:
        _min, _max = d_img.min(), d_img.max()
        if _max == _min:
            img_norm = d_img - _max
        else:
            img_norm = (d_img - _min) * 255. / (_max - _min)
        img_norm = img_norm.astype('uint8')
    skimage.io.imsave(fn, img_norm, check_contrast=False)

def scale2uint8(_img):
    _min, _max = _img.min(), _img.max()
    if _max == _min:
        _img_s = _img - _max
    else:
        _img_s = (_img - _min) * 255. / (_max - _min)
    _img_s = _img_s.astype('uint8')
    return _img_s

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
