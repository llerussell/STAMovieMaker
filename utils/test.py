# STA Movie Maker
# Lloyd Russell 2017

import sys
import os
import ctypes
import json
import numpy as np
import difflib
import glob
import colorsys
import scipy
from scipy import io as sio
from skimage.external import tifffile
import time
import sta, PrairieLink, paq2py, ThorLink
from skimage import exposure
import tensorflow as tf



moviePath = 'E:/Analysis/20171115/20171113_L438_t-001_REG.bin'
movie = PrairieLink.ReadRawFile(moviePath)

ds_movie = sta.downsample(movie, 20)
# tifffile.imsave(moviePath.replace('.bin', '_DS.tif'), ds_movie)
corr_img = sta.makeCorrImg(ds_movie, 8)
# corr_img = corr_img * 65535
tifffile.imsave(moviePath.replace('.bin', '_CORR.tif'), corr_img.astype(np.float32))