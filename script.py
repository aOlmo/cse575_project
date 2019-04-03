import os
import cv2
import glob
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import imageio

__TEST_IMG__ = "data/dog.jpg"
__RESULTS_FOLDER__ = "results/dog_black"
__IMGS_FOLDER__ = "data/"

<<<<<<< HEAD
try:
    os.mkdir(__RESULTS_FOLDER__)
except:
    pass
=======
>>>>>>> 3fceb217f009ffdca4375f0f7f7bb02895cb2809

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
def get_rectangle_mask(mask):
    val = 255
    cv2.rectangle(mask, (221, 100), (130, 137), (val, val, val), -1)
    cv2.rectangle(mask, (0, 210), (100, 120), (val, val, val), -1)

    return mask


def get_random_rectangle_mask(mask):
    val = 255
    x1, y1 = np.random.randint(200, 257, size=2)  # randint(low 'inclusive' , high 'exclusive', size=None)
    offset = np.random.randint(40, min(x1, y1, 100))  # force larger rectangles by having value > 40 but < 120
    cv2.rectangle(mask, (x1, y1), (x1 - offset, y1 - offset), (val, val, val), -1)

    x2, y2 = np.random.randint(50, min(x1 - offset, y1 - offset), size=2)  # force second rectangle in top left corner above first rectangle
    offset = np.random.randint(40, min(x2, y2))
    cv2.rectangle(mask, (x2, y2), (x2 - offset, y2 - offset), (val, val, val), -1)

    return mask


def get_circle_mask(mask):
    val = 255  # white = rgb(255, 255, 255)
    cv2.circle(mask, (70, 200), 30, (val, val, val), -1)  # circle(img, point center, int radius, color rgb, thickness)
    cv2.circle(mask, (200, 60), 50, (val, val, val), -1)

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

    for i, file in enumerate(glob.glob(__IMGS_FOLDER__ + "*.*")):
        name, ext = os.path.splitext(file.split("/")[-1])
        img = get_img(file)
        img = cv2.resize(img, (256, 256))

        mask_zeros = np.zeros_like(img)
        # aux = apply_blur(img, intensity=90)

        # aux = np.zeros((256, 256, 3), 'uint8')
        # aux[..., 0] = 0
        # aux[..., 1] = 0
        # aux[..., 2] = 0

        mask_zeros = get_mask(mask_zeros)
        img_with_blurs = np.where(mask_zeros == np.array([255, 255, 255]), aux, img)

        imgs_side_2_side = np.hstack((img, img_with_blurs))

        # Save image
        imageio.imsave(__RESULTS_FOLDER__+"/"+str(i+1)+ext, imgs_side_2_side)

        # TODO: Save mask too
