# -*- coding: utf-8 -*-
"""Batch_Color_Normalization_for_CRC_Data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pOV60rNqyx-rVAOMf6oTO3DxuDuZfaMV
"""

pip install color_transfer

import cv2
import glob
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from color_transfer import color_transfer

from google.colab import drive
drive.mount('/content/drive')
folders = glob.glob('/content/drive/MyDrive/Paper_Data/images')
imagenames_list = []
for folder in folders:
    for f in glob.glob(folder+'/*'):
        imagenames_list.append(f)

def resize_image(image, width = 250):
    r = width/float(image.shape[1])
    dim = (width, int(image.shape[0]*r))
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized

def image_stats(image):
    # Compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (l_mean, l_std) = (l.mean(), l.std())
    (a_mean, a_std) = (a.mean(), a.std())
    (b_mean, b_std) = (b.mean(), b.std())
    # return the color statistics
    return (l_mean, l_std, a_mean, a_std, b_mean, b_std)

def _min_max_scale(arr, new_range=(0, 255)):
    mn = arr.min()
    mx = arr.max()
    # check if scaling needs to be done to be in new_range
    if mn < new_range[0] or mx > new_range[1]:
		# perform min-max scaling
        scaled = (new_range[1] - new_range[0]) * (arr - mn) / (mx - mn) + new_range[0]
    else:
		# return array if already in range
	    scaled = arr

    return scaled

def _scale_array(arr, clip=True):
	if clip:
		scaled = np.clip(arr, 0, 255)
	else:
		scale_range = (max([arr.min(), 0]), min([arr.max(), 255]))
		scaled = _min_max_scale(arr, new_range=scale_range)

	return scaled

def color_transfer(source, target, clip=True):

    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

    (l, a, b) = cv2.split(source)
    q = (lStdTar-lStdSrc)/lStdTar

    if q>0:
        l = lMeanSrc+((l-lMeanSrc)*(1+q))
    else:
        l = lMeanSrc+(l-lMeanSrc)*(1+0.05)

        a = aMeanTar+(a-aMeanSrc)
        b = bMeanTar+(b-bMeanSrc)

        l = _scale_array(l, clip=clip)
        a = _scale_array(a, clip=clip)
        b = _scale_array(b, clip=clip)

	# merge the channels together and convert back to the RGB color
	# space, being sure to utilize the 8-bit unsigned integer data
	# type
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

	# return the color transferred image
    return transfer





source_images = []
transferred_images = []

for image in imagenames_list:
    source = cv2.imread(image)
    if source is None:
        print(f"Could not read the image: {image}")
        continue

    source = resize_image(source)
    source_images.append(source)

    target_path = r"images/Patient_012_01_Normal.png"
    target = cv2.imread(target_path)
    if target is None:
        print(f"Could not read the target image: {target_path}")
        continue

    target = resize_image(target)
    transferred = color_transfer(source, target)
    transferred = resize_image(transferred)

    output_path = f"/content/drive/MyDrive/Paper_Data/output(PM)/output_{image.split('/')[-1]}"
    output = cv2.imwrite(output_path, transferred)
    transferred_images.append(output)

import cv2
from PIL import Image
import glob

for f in glob.glob('/content/drive/MyDrive/Paper_Data/images/*'):
    img = cv2.imread(f)
    im_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    im_hsv_pil = Image.fromarray(im_hsv)
    im_hsv_pil.save('/content/drive/MyDrive/Paper_Data/RGB2HSV/%s.png' % f.split("/")[-1])

    im_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    im_lab_pil = Image.fromarray(im_lab)
    im_lab_pil.save('/content/drive/MyDrive/Paper_Data/RGB2LAB/%s.png' % f.split("/")[-1])

    ycbcr_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycbcr_image_pil = Image.fromarray(ycbcr_image)
    ycbcr_image_pil.save('/content/drive/MyDrive/Paper_Data/ycbcr_image/%s.png' % f.split("/")[-1])
