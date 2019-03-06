import os
import cv2
import glob
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

__TEST_IMG__ = "data/dog.jpg"
__RESULTS_FOLDER__ = "results/"
__IMGS_FOLDER__ = "data/"

def display_img(rgb_img):
    plt.imshow(rgb_img)
    # to hide tick values on X and Y axis
    plt.xticks([]), plt.yticks([])
    plt.show()


def get_sample_img():
    bgr_img = cv2.imread(__TEST_IMG__)
    # get bgr and switch to rgb
    b, g, r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r, g, b])
    return rgb_img

def apply_mask(mask, img):
    cv2.rectangle(mask, (265, 17), (380, 137), 0, -1)
    cv2.rectangle(mask, (0, 280), (100, 320), 0, -1)

    # Apply the mask and return the result
    return cv2.bitwise_and(img, mask)


def get_img(img):
    bgr_img = cv2.imread(img)
    # get bgr and switch to rgb
    b, g, r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r, g, b])
    return rgb_img

if __name__ == '__main__':

    for file in glob.glob(__IMGS_FOLDER__+"*.*"):
        name, ext = os.path.splitext(file.split("/")[-1])

        img = get_img(file)
        # When performing the bitwise and we need to have 0s in the mask
        # and the original RGB value for the other regions, thus in a
        # Bitwise operation, we need whatever RGB value is and 2)11111111 = 10)255
        mask = np.full_like(img, 255)

        # Apply the mask and display image
        masked_img = apply_mask(mask, img)

        # Save image
        scipy.misc.imsave(__RESULTS_FOLDER__+name+"_masked"+ext, masked_img)

        # TODO: Save mask too
        display_img(masked_img)

