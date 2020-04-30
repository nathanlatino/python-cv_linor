import numpy as np
from cv2 import cv2

def roi(img, vertices):
	# blank mask:
	mask = np.zeros_like(img)
	# fill the mask
	cv2.fillPoly(mask, vertices, 255)
	# now only show the area that is the mask
	masked = cv2.bitwise_and(img, mask)
	return masked

def white_mask(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	low_white = np.array([100, 0, 210])
	upper_white = np.array([250, 255, 255])

	mask = cv2.inRange(hsv, low_white, upper_white)
	return mask

def morpho_process(img):
	# delete lines
	# img2 = morph(img, (4,4))
	# strengthen intersections
	img2 = morph(img, (9, 9), 3, mode='d')
	img2 = morph(img2, (4, 4), 2)
	# close remaining blobs
	img2 = morph(img2, (12, 12))
	img2 = morph(img2, (12, 12), mode='d')
	img2 = morph(img2, (12, 12))
	img2 = morph(img2, (3, 3))
	return img2

def morph(img, size=(3, 3), iteration=1, mode='e'):
	kernel = np.ones(size, np.uint8)
	if mode == 'e':
		return cv2.erode(img, kernel, iterations=iteration)
	else:
		return cv2.dilate(img, kernel, iterations=iteration)
