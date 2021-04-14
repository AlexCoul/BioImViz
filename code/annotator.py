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
import potrace
from skimage.color import rgb2gray

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
# vectorisation du mask en .svg

import numpy as np
import potrace

# Make a numpy array with a rectangle in the middle
data = np.zeros((32, 32), np.uint32)
data[8:32-8, 8:32-8] = 1

# Create a bitmap from the array
bmp = potrace.Bitmap(grayscale)

# Trace the bitmap to a path
path = bmp.trace()

# pas de fonction pour exporter

# %%
type(path)

# %%
im.shape

# %%
grayscale = rgb2gray(im)

plt.imshow(grayscale)

# %%
grayscale.shape

# %%
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM

drawing = svg2rlg("file.svg")
renderPDF.drawToFile(drawing, "file.pdf")
renderPM.drawToFile(drawing, "file.png", fmt="PNG")

# %%
import numpy as np
import matplotlib.pyplot as plt

from skimage import measure


# ouverture du mask_cell

mask_im = Image.open(fold_im + '/processed/' + name_im + '_mask_cells.tiff')
mask_im = np.array(mask_im)

plt.imshow(mask_im)

# Find contours at a constant value of 0.8
contours = measure.find_contours(mask_im, 1)

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(mask_im, cmap=plt.cm.gray)

for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=3)

plt.show()

# %%
annots = {
    1: {
        "x":contours[0][:,0],
        "y":contour[0][:,1],
    },
    2: {
        "x":contour[1][:,0],
        "y":contour[1][:,1],
    },
}

# %%
annots = {}
for i, cont in enumerate(contours):
    annots[i+1] = {
        "x":cont[:,0],
        "y":cont[:,1], 
    }

# %%
# lire l'image avec chaque annotation une par une puis la compiler

for i in np.unique(mask_im)[1:]:
    #fait le contour de l'annotation de valeur i
    binarized = mask_im==i
    plt.imshow(binarized)
    plt.show()

# %%
mask_im==i

# %%
np.unique(mask_im)

# %%
