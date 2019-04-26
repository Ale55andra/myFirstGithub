import numpy as np
import cv2
import imutils
import random as rng

from functions import (resize, contours, findPage, transform, sharp, equalize, focus, show, contrast, checkBrightness)



# Import Image and resize
image = cv2.imread('flower.jpg')
image = resize(image)
imageOriginal = image.copy()

# Checks the brightness of the image and adjusts
#If the image is too blurry, improve contrast by equalizing histogram channels
image = equalize(image)


# Find the item you want to enlarge.  If it is not correct, adjust the k value to an odd number from 0-9.
k = 5
contours = contours(image,k)
rect, drawing = findPage(contours,imageOriginal)
name = 'Box Drawn'
show(drawing,name)


# Warp the boxed item to full screen.  You can play with focusing/sharpening the image based on your needs.

warp = transform(imageOriginal, rect)
warp = checkBrightness(warp)



# Focus/Sharpen the scanned photo. Change the alpha value to adjust the level of focus in the photo
alpha = 4
warp = focus(warp, alpha)
warp = sharp(warp)


name = 'Scanned Photo'
show(warp,name)
