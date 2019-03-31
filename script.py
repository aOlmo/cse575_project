import os
import cv2
import glob
from scipy import misc
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

# TODO: Create more masks
def get_mask(mask):
    val = 255
    cv2.rectangle(mask, (221, 17), (321, 137), (val, val, val), -1)
    cv2.rectangle(mask, (0, 210), (100, 120), (val, val, val), -1)

    return mask

def apply_blur(img, intensity=35):
    blurred_image = cv2.blur(img, (intensity, intensity))
    return blurred_image


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
        img = cv2.resize(img, (256, 256))

        mask_zeros = np.zeros_like(img)
        blurred_img = apply_blur(img)

        mask_zeros = get_mask(mask_zeros)
        img_with_blurs = np.where(mask_zeros == np.array([255, 255, 255]), blurred_img, img)

        numpy_horizontal = np.hstack((img, img_with_blurs))

        display_img(numpy_horizontal)

        # Save image
        misc.imsave(__RESULTS_FOLDER__+name+"_masked"+ext, img_with_blurs)

        # TODO: Save mask too
