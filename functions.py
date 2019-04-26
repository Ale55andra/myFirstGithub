import numpy as np
import cv2
import imutils
import random as rng


def resize(image):

    # resizes images to a reasonable display size
    image = image.astype('float32')
    ratio = image.shape[0] / 300.0 #uses the ratio of the image to resize
    orig = image.copy()
    image = cv2.normalize(image,None,0,255,cv2.NORM_MINMAX)
    image = imutils.resize(image, height = 300)
    return image

def contours(img,k):

    """
    First I blur the image to reduce noisen and convert to proper format for inputs
    in the following functions.
    """

    img = img.astype('float32')
    if (k == 0):
        img =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.GaussianBlur(img,(k,k),0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_image = img.astype(np.uint8)

    """
    Second, I use canny to detect the corners of the image, which produces a binary
    image.
    """

    corners = cv2.Canny(blurred_image,1,250)
    contours = cv2.findContours(corners.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(contours)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
    return cnts

def findPage(contours,img):
    """
    I used http://www.swarthmore.edu/NatSci/mzucker1/opencv-2.4.10-docs/doc/tutorials/imgproc/shapedescriptors/bounding_rects_circles/bounding_rects_circles.html
    to produce parts of this code.

    This function creates a bounding box around the page you want to scan.

    I first used the contours to create approximate polygons (closed loop).

    Second, I created a bounding rectangle around all the polygons that I found.

    Lastly, I calculated the area of the bounding rectangles and selected the rectangle
    with the largest area as my page that I wanted to enlarge/scan.

    The output is the (x,y) coordinates of the upper left corner, the width, and
    height of the bounding rectangle (rectangle) and the image with the bounding
    rectangle drawn on it.

    """
    img = img.astype('float32')

    # Find bounding Rectangles from the contours
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, False)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
    drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    imgCopy = img.copy()

    # Determine the area of each of the bounding rectangles and find the max
    area = np.zeros((len(boundRect),1))
    for i in range(len(boundRect)):
        area[i] = (int(boundRect[i][2]))*(int(boundRect[i][3]))

    max = np.argmax(area)
    rectangle = boundRect[max]

    # Draw the rectangle with the largest area onto the image
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    cv2.rectangle(imgCopy, (int(boundRect[max][0]), int(boundRect[max][1])), \
        (int(boundRect[max][0]+boundRect[max][2]), int(boundRect[max][1]+boundRect[max][3])), color, 2)

    return rectangle, imgCopy

def transform(image, rectangle):

    """
    I used https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/ as a base for this part of the code.
    This function finds coordinates of the rectangle from the output of the boudingRect() cv2 function
    (the (x,y) coordinate of the top left corner of the rectangle, the width, and the height)
    With these coordinates, the scanned document is warped to fill the page.

    """
    image = image.astype('float32')

    factor = 1.9

    #find corners of rectangle
    x = int(rectangle[0])
    y = int(rectangle[1])
    w = int(rectangle[2])
    h = int(rectangle[3])

    tl = np.array([x,y])
    tr = np.array([x+w,y])
    br = np.array([x+w,y+h])
    bl = np.array([x,y+h])

    rect = np.array([tl,tr,br,bl])

    #creates the matrix to place the corners
    dst = np.zeros((4,2), dtype ="float32")

    #Multiplies the height and width by a factor to determine the size/dimensions of the final image
    dst[0] = np.array([0,0]) #top left
    dst[1] = np.array([w*factor,0]) #top right
    dst[2] = np.array([w*factor,h*factor]) #bottow right
    dst[3] = np.array([0,h*factor]) #bottom left

    dst = dst.astype('float32')

    maxWidth = int(w*factor)
    maxHeight = int(h*factor)

    rect = rect.astype('float32')
    M = cv2.getPerspectiveTransform(rect,dst)
    warp = cv2.warpPerspective(image,M,(maxWidth, maxHeight))
    return warp

def sharp(img):
    """
    This function uses a sharpening kernel that is convolved over the image.
    """
    img = img.astype('uint8')
    r, c, ch = img.shape
    kernel= np.array([[0,-.5,0],
                        [-.5, 3,-.5],
                        [0,-.5,0]])


    img = cv2.filter2D(img,-1,kernel)
    img = img.astype('uint8')

    return img

def equalize(img):
    """
    This function equalizes the histogram in the bgr channels to improve contrast.
    """
    img = img.astype('uint8')
    img[:,:,0] = cv2.equalizeHist(img[:,:,0])
    img[:,:,1] = cv2.equalizeHist(img[:,:,1])
    img[:,:,2] = cv2.equalizeHist(img[:,:,2])

    return(img)

def focus(img, alpha):
    """
    This function is based off the equation for focus:
    focused image = image + alpha * (image - gaussian blur)
    """
    img = img.astype('uint8')
    gauss = cv2.GaussianBlur(img,(3,3),0)
    addToSharpen = cv2.multiply(alpha,cv2.subtract(img,gauss))
    img = cv2.add(img,addToSharpen)

    return img

def show(img,name):
    img = img.astype('uint8')
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def contrast(img, contrast):
    img = img.astype('float32')
    img = np.multiply(img,contrast)

    return img

def checkBrightness(img):
    #This function corrects the brightness of the image to be between 75-200.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean, x,y,z = cv2.mean(gray)

    if mean > 200:
        g = mean - (200-mean)
        img = cv2.add(img,g)
    elif mean < 75:
        g = (75-mean)+mean
        img = cv2.add(img,g)

    return img
