#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

Functions for cell detecting and ROI extraction.
Functions for embrane detection and membrane regions extraction with hysteresis filter.
Optimysed for widefield neuron image.

"""

import os
import logging

import numpy as np
import numpy.ma as ma

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage.external import tifffile
from skimage import filters
from skimage import measure
from skimage import segmentation

from scipy import ndimage as ndi
from scipy.ndimage import measurements as msr



def backRm(img, edge_lim=20, dim=3):
    """ Background extraction in TIFF series

    For confocal Z-stacks only!
    dem = 2 for one frame, 3 for z-stack

    """
    if dim == 3:
        edge_stack = img[:,:edge_lim,:edge_lim]
        mean_back = np.mean(edge_stack)

        logging.info('Mean background, {} px region: {:.3f}'.format(edge_lim, mean_back))

        img_out = np.copy(img)
        img_out = img_out - mean_back
        img_out[img_out < 0] = 0

        return img_out
    elif dim == 2:
        edge_fragment = img[:edge_lim,:edge_lim]
        mean_back = np.mean(edge_fragment)

        logging.info(f"Mean background, {edge_lim} px region: {mean_back}")

        img = np.copy(img)
        img = img - mean_back
        img[img < 0] = 0

        return img


def de_bleach(img_series, base_num=0):
    """ Wide-field image series bleaching correction.
    Base on first frame apply pixels values correction.

    """
    base_frame = img_series[base_num,:,:]
    corr_series = [frame * np.true_divide(base_frame, frame, out=np.zeros_like(base_frame), where=frame!=0) for frame in img_series]
    corr_err =  np.std([np.sum(frame) for frame in corr_series]) / np.mean([np.sum(frame) for frame in corr_series])  # mean of series frames sum / sd of series frame sum
    logging.info(f'Mean correction deviation={round(corr_err*100, 3)}%') 
    return corr_series


def hyst_mask(img, high=0.8, low=0.2, sigma=3):
    """ Function for neuron region detection with hysteresis threshold algorithm.

    img - input image with higest intensity;
    gen_high - float,  general upper threshold for hysteresis algorithm (percentage of maximum frame intensity);
    sigma - int, sd for gaussian filter.

    Returts cell boolean mask for input frame.

    """
    img_gauss = filters.gaussian(img, sigma=sigma)
    num = 10
    while num > 1:
        mask = filters.apply_hysteresis_threshold(img_gauss,
                                                  low=np.max(img_gauss)*high,
                                                  high=np.max(img_gauss)*low)
        a, num = ndi.label(mask)
        low -= 0.01
    logging.info(f"Lower limit for hystMask={round(low, 2)}")
    return mask


def series_derivate(series, mask, sigma=4, kernel_size=3,  sd_area=50, sd_tolerance=False, left_w=1, space_w=0, right_w=1, output_path=False):
    """ Calculation of derivative image series (difference between two windows of interes).

    """
    trun = lambda k, sd: (((k - 1)/2)-0.5)/sd  # calculate truncate value for gaussian fliter according to sigma value and kernel size
    gauss_series = np.asarray([filters.gaussian(series[i], sigma=sigma, truncate=trun(kernel_size, sigma)) for i in range(np.shape(series)[0])])

    logging.info(f'Derivate sigma={sigma}')

    der_series = []
    for i in range(np.shape(gauss_series)[0] - (left_w+space_w+right_w)):
        der_frame = np.mean(gauss_series[i+left_w+space_w:i+left_w+space_w+right_w], axis=0) - np.mean(gauss_series[i:i+left_w], axis=0)
        if sd_tolerance:
            der_sd = np.std(der_frame[:sd_area, sd_area])
            der_frame[der_frame > der_sd * sd_tolerance] = 1
            der_frame[der_frame < -der_sd * sd_tolerance] = -1
        der_series.append(ma.masked_where(~mask, der_frame))    
    logging.info(f'Derivative series len={len(der_series)} (left WOI={left_w}, spacer={space_w}, right WOI={right_w})')

    if output_path:
        save_path = f'{output_path}/blue_red'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        norm = lambda x, min_val, max_val: (x-min_val)/(max_val-min_val)  # normilize derivate series values to 0-1 range
        vnorm = np.vectorize(norm)

        for i in range(len(der_series)):
            frame = der_series[i]
            # frame = vnorm(raw_frame, np.min(der_series), np.max(der_series))

            plt.figure()
            ax = plt.subplot()
            img = ax.imshow(frame, cmap='bwr')
            img.set_clim(vmin=np.min(der_series), vmax=np.max(der_series)) 
            div = make_axes_locatable(ax)
            cax = div.append_axes('right', size='3%', pad=0.1)
            plt.colorbar(img, cax=cax)
            ax.text(10,10,i+1,fontsize=10)
            ax.axis('off')

            plt.savefig(f'{save_path}/frame_{i+1}.png')
            logging.info(f'Derivate frame {i+1} saved!')
            plt.close('all')
        return np.asarray(der_series)
    else:
        return np.asarray(der_series)


def series_point_delta(series, mask, mask_series=False, baseline_frames=3, delta_min=1, delta_max=-1, sigma=4, kernel_size=5, output_path=False):
    trun = lambda k, sd: (((k - 1)/2)-0.5)/sd  # calculate truncate value for gaussian fliter according to sigma value and kernel size
    img_series = np.asarray([filters.gaussian(series[i], sigma=sigma, truncate=trun(kernel_size, sigma)) for i in range(np.shape(series)[0])])

    baseline_img = np.mean(img_series[:baseline_frames,:,:], axis=0)

    delta = lambda f, f_0: (f - f_0)/f_0 if f_0 > 0 else f_0 
    vdelta = np.vectorize(delta)

    if mask_series:
        delta_series = [ma.masked_where(~mask_series[i], vdelta(img_series[i], baseline_img)) for i in range(len(img_series))]
    else:
        delta_series = [ma.masked_where(~mask, vdelta(i, baseline_img)) for i in img_series]

    if output_path:
        save_path = f'{output_path}/delta_F'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i in range(len(delta_series)):
            frame = delta_series[i]

            plt.figure()
            ax = plt.subplot()
            img = ax.imshow(frame, cmap='jet')
            img.set_clim(vmin=delta_min, vmax=delta_max) 
            div = make_axes_locatable(ax)
            cax = div.append_axes('right', size='3%', pad=0.1)
            plt.colorbar(img, cax=cax)
            ax.text(10,10,i+1,fontsize=10)
            ax.axis('off')

            file_name = save_path.split('/')[-1]
            plt.savefig(f'{save_path}/{file_name}_frame_{i+1}.png')
            logging.info(f'Delta F frame {i+1} saved!')
            plt.close('all')
        return np.asarray(delta_series)
    else:
        return np.asarray(delta_series)


if __name__=="__main__":
    pass


# That's all!
