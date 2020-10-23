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


img_series = tifffile.imread(os.path.join(sys.path[0], 'demo_data/hpca_cfp.tif'))
img_series = d.back_rm(img_series)
img_series = img_series[:,:,:550]


mask = d.hyst_mask(img_series[0], low=0.2)
# thresh = filters.threshold_otsu(img_series[0])
# mask = img_series[0] > thresh

# skel, dist = medial_axis(mask, return_distance=True, )
# dist_plot = dist * skel
skel = skeletonize(mask, method='lee')

diff_series = d.s_derivate(img_series, mask,
	                      sigma=4, mean_win=2, mean_space=0, mode='binn')




num = 6

ax2 = plt.subplot(121)
img2 = ax2.imshow(img_series[num])
dvdr2 = make_axes_locatable(ax2)
cax2 = dvdr2.append_axes('right', size='2%', pad=0.05)
plt.colorbar(img2, cax=cax2)


ax1 = plt.subplot(122)
ax1.imshow(mask)
# ax1.contour(mask, size=0.5, colors='b')

# ax3 = plt.subplot(122)
# img3 = ax3.imshow(diff_series[num],
# 	       vmin=-1, vmax=1,
# 	       cmap='bwr')
# dvdr3 = make_axes_locatable(ax3)
# cax3 = dvdr3.append_axes('right', size='2%', pad=0.05)
# plt.colorbar(img3, cax=cax3)
plt.show()



# # save series
# a = 1
# for frame in diff_series:
# 	plt.figure()
# 	ax = plt.subplot()
# 	img = ax.imshow(frame, cmap='bwr')
# 	ax.text(20,20,a,fontsize=18)
	# ax.axis('off')
	# plt.savefig('frame_{}.png'.format(a))

	# logging.info('Frame {} saved!'.format(a))
	# a += 1


# # animation
# fig = plt.figure()
# img = plt.imshow(diff_series[0], cmap='seismic')
# def ani(i):
# 	img.set_array(diff_series[i])
# 	return img,
# ani = anm.FuncAnimation(fig, ani, interval=10, frames=len(diff_series))
# plt.show()
