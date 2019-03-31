import os
import cv2
import glob
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


# TODO: Create a set of different masks
def apply_mask(mask, img, opposite=False):
    val = 0 if not opposite else 255

    cut_img = img[17:137, 265:380]

    blurred_patch = apply_blur(cut_img)
    img[17:137, 265:380] = 0
    img[17:137, 265:380] = blurred_patch

    display_img(img)

    cv2.rectangle(mask, (265, 17), (380, 137), (val, val, val), -1)
    cv2.rectangle(mask, (0, 280), (100, 320), (val, val, val), -1)

    display_img(mask)
    exit()

    # Apply the mask and return the result
    return cv2.bitwise_and(img, mask)


def apply_blur(img):
    blurred_image = cv2.blur(img, (40,40))
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
        # When performing the bitwise and we need to have 0s in the mask
        # and the original RGB value for the other regions, thus in a
        # Bitwise operation, we need whatever RGB value is and 2)11111111 = 10)255
        mask_full = np.full_like(img, 255)
        mask_zeros = np.full_like(img, 0)

        # To apply blur:
        # ------------------------------------------------------------------
        # 1) Get image twice and remove those regions of interest from one of them (values = 0)
        # 2) In the other one, have exactly the opposite, only the parts of the image we are interested in
        # 3) Apply blur to the second one
        # 4) Add both together
        # ------------------------------------------------------------------

        # Apply the mask and display image
        masked_img = apply_mask(mask_full, img)
        opposite_img = apply_mask(mask_zeros, img, True)

        # Apply blur to opposite img
        #TODO: solve borders problem:
        # it could be because when adding, the values around the blurred borders are not 0
        # (but they should)
        blurred_patches = apply_blur(opposite_img)
        display_img(blurred_patches)
        final_img = masked_img + blurred_patches
        display_img(final_img)

        # Save image
        # scipy.misc.imsave(__RESULTS_FOLDER__+name+"_masked"+ext, masked_img)

        # TODO: Save mask too
