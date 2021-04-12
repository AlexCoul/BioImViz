#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
# %qtconsole

# %%
"""
Created on Tue Nov 24 18:45:14 2020

@author: alexis
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import napari
import tifffile
from pathlib import Path
from PIL import Image

fold_im = "/home/vladimir/Documents/Projets/to_annotate/Brightfield Confluence/Faible confluence"
name_im = "AH-IPC Mock 2020-10-23 Well B2 Field 2 Time 1 - faible confluence.TIFF"

path_im = fold_im + "/" + name_im

# %%
# affichage de l'image avec son titre

im = Image.open(path_im)
im = np.array(im)

plt.imshow(im)
plt.title(name_im)
plt.show()

# %%
# conversion en 16 bit

im = Image.open(path_im)
im = np.array(im, dtype=np.uint16)
im *= 256

# %%
# ouverture de toutes les images d'un dossier

fold = Path(fold_im)

for img in fold.iterdir() :
    im = Image.open(img)
    im = np.array(im)
    plt.imshow(im)
    plt.show()

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
# %%
# ouverture de napari et création du mask cell

label_cells = np.zeros_like(im[:,:,0])
with napari.gui_qt():
    viewer = napari.view_image(im, rgb=True, name=name_im)
    viewer.add_labels(label_cells, name='mask_cells')

# %%
# sauvegarde du mask en fonction du nom de l'image d'origine

if Path(fold_im + '/processed').exists() :
    tifffile.imwrite(fold_im + '/processed/' + name_im + '_mask_cells.tiff', viewer.layers['mask_cells'].data)
else :
    new_fold = Path(fold_im + '/processed')
    new_fold.mkdir()
    tifffile.imwrite(fold_im + '/processed/' + name_im + '_mask_cells.tiff', viewer.layers['mask_cells'].data)

# %%
