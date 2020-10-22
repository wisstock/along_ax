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
from skimage.external import tifffile

sys.path.append('modules')
import diff as d


FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=FORMAT)

plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#272b30'
plt.rcParams['image.cmap'] = 'inferno'


img_series = tifffile.imread(os.path.join(sys.path[0], 'data/two/hpca_cfp.tif'))
img_series = d.backRm(img_series)


mask = d.hystMask(img_series[0])
diff_series = d.sDerivate(img_series, mask,
	                      sigma=4, mean_win=2, mean_space=0, mode='binn')


num = 6

ax2 = plt.subplot(121)
img2 = ax2.imshow(img_series[num])
dvdr2 = make_axes_locatable(ax2)
cax2 = dvdr2.append_axes('right', size='2%', pad=0.05)
plt.colorbar(img2, cax=cax2)

ax3 = plt.subplot(122)
img3 = ax3.imshow(diff_series[num],
	       vmin=-1, vmax=1,
	       cmap='bwr')
dvdr3 = make_axes_locatable(ax3)
cax3 = dvdr3.append_axes('right', size='2%', pad=0.05)
plt.colorbar(img3, cax=cax3)

plt.show()



# # save series
# a = 1
# for frame in diff_series:
# 	plt.figure()
# 	ax = plt.subplot()
# 	img = ax.imshow(frame, cmap='bwr')

# 	ax.text(20,20,a,fontsize=18)
# 	# rect = patches.Rectangle((0,0),100,100,linewidth=1,edgecolor='w',facecolor='k')
# 	# ax.add_patch(rect)
	

# 	ax.axis('off')
# 	plt.savefig('frame_{}.png'.format(a))

# 	logging.info('Frame {} saved!'.format(a))
# 	a += 1


# # animation
# fig = plt.figure()
# img = plt.imshow(diff_series[0], cmap='seismic')
# def ani(i):
# 	img.set_array(diff_series[i])
# 	return img,
# ani = anm.FuncAnimation(fig, ani, interval=10, frames=len(diff_series))

# plt.show()
