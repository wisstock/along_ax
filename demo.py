#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

Demo for wide-field image translocation analysis.

"""

import sys
import os
import logging

import numpy as np
import numpy.ma as ma

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage import filters
from skimage.morphology import medial_axis, skeletonize
from skimage.external import tifffile

sys.path.append('modules')
import diff as d


FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=FORMAT)

plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#272b30'
plt.rcParams['image.cmap'] = 'inferno'

res_path = os.path.join(sys.path[0], 'img')


img_series = tifffile.imread(os.path.join(sys.path[0], 'demo_data/hpca_cfp.tif'))
img_series = img_series[:,:,:550]
img_series = d.back_rm(img_series)
img_series = d.de_bleach(img_series)

mask = d.hyst_mask(np.mean(img_series, 0), low=0.2)

# thresh = filters.threshold_otsu(img_series[0])
# mask = img_series[0] > thresh


# skel, dist = medial_axis(mask, return_distance=True, )
# dist_plot = dist * skel
# skel = skeletonize(mask, method='lee')

diff_series = d.series_derivate(img_series, mask,
	                            sigma=5, kernel_size=9,
	                            sd_tolerance=5,
                                left_w=2, space_w=0, right_w=2, output_path=res_path)

delta_series = d.series_point_delta(img_series, mask,
	                                baseline_frames=3, delta_min=.2, delta_max=-.2,
	                                sigma=5, kernel_size=9,
	                                output_path=res_path)

# save series
for i in range(len(img_series)):
	frame = img_series[i]
	plt.figure()
	ax = plt.subplot()
	img = ax.imshow(frame)
	ax.imshow(mask, alpha=.10)
	img.set_clim(vmin=np.min(img_series), vmax=np.max(img_series)) 
	ax.text(10,10,i+1,fontsize=10)
	ax.axis('off')
	plt.savefig(f'{res_path}/corr/frame_{i+1}.png')
	logging.info(f'Frame {i+1} saved!')
	plt.close('all')





# num = 6

# ax2 = plt.subplot(121)
# img2 = ax2.imshow(img_series[num])
# dvdr2 = make_axes_locatable(ax2)
# cax2 = dvdr2.append_axes('right', size='2%', pad=0.05)
# plt.colorbar(img2, cax=cax2)


# ax1 = plt.subplot(122)
# ax1.imshow(mask)
# ax1.contour(mask, size=0.5, colors='b')

# ax3 = plt.subplot(122)
# img3 = ax3.imshow(diff_series[num],
# 	       vmin=-1, vmax=1,
# 	       cmap='bwr')
# dvdr3 = make_axes_locatable(ax3)
# cax3 = dvdr3.append_axes('right', size='2%', pad=0.05)
# plt.colorbar(img3, cax=cax3)
# plt.show()




# # animation
# fig = plt.figure()
# img = plt.imshow(diff_series[0], cmap='seismic')
# def ani(i):
# 	img.set_array(diff_series[i])
# 	return img,
# ani = anm.FuncAnimation(fig, ani, interval=10, frames=len(diff_series))
# plt.show()
