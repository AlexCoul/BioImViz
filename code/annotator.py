#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 18:45:14 2020

@author: alexis
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import napari


path_im = "/home/alexis/Documents/Pro/Post-doc/Projects/Wide_Field_Segmentation/data/raw/Agathe/Collection image OPERETTA/Propagation/Brightfield/RK13 MOI 10 2020-10-30 Well B4 Field 3 Time 43 - brightfield propagation.tiff"

# %%
im = io.imread(path_im)
# %%
import javabridge
import bioformats
javabridge.start_vm(class_path=bioformats.JARS)

im = bioformats.load_image(path_im)

javabridge.kill_vm()

# %%

from aicsimageio import AICSImage, imread, imwrite

# Get an AICSImage object
im = AICSImage(path_im)
im.data  # returns 6D STCZYX numpy array
im.dims  # returns string "STCZYX"
im.shape  # returns tuple of dimension sizes in STCZYX order
im.get_image_data("CZYX", S=0, T=0)  # returns 4D CZYX numpy array

# Get 6D STCZYX numpy array
data = imread(path_im)
# %%


import tifffile
im = tifffile.imread(path_im)

# %%
from PIL import Image
im = Image.open(path_im)
im = np.array(im)
# %% Extract green channel from composite image
from skimage.color import rgb2hsv
plt.figure()
plt.imshow(im)

hsv_img = rgb2hsv(im)
hue_img = hsv_img[:, :, 0]
value_img = hsv_img[:, :, 2]

plt.imshow(hue_img)
plt.colorbar()

plt.figure()
plt.hist(hue_img.ravel(), bins=50)

plt.figure()
plt.imshow(value_img)

select = hue_img < 0.10


# %%

label_cells = np.zeros_like(im[:,:,0])
with napari.gui_qt():
    viewer = napari.view_image(im, rgb=True)
    viewer.add_labels(label_cells, name='cells')


def export_annotations():
    tifffile.imwrite('/home/alexis/Documents/Pro/Post-doc/Projects/napari/data/processed/annotations-cells.tif', viewer.layers['cells'].data)
