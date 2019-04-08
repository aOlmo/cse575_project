import sys
import cv2
import glob
import numpy as np
import argparse
import matplotlib.pyplot as plt

from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_mse as mse


__TEST_FOLDER__ = "results/test/"

def display_img(rgb_img):
    plt.imshow(rgb_img)
    plt.xticks([]), plt.yticks([])
    plt.show()

def get_avg_metric(images, metric):
    height, width = images.shape[1], images.shape[2]

    width_cutoff = width // 2
    s1 = images[..., :width_cutoff, :]
    s2 = images[..., width_cutoff:, :]

    sum = 0
    for left, right in zip(s1, s2):
        if (metric == "ssim"):
            sum += ssim(left, right, data_range=right.max() - right.min(), multichannel=True)
        elif (metric == "psnr"):
            sum += psnr(left, right)
        elif (metric == "mse"):
            sum += mse(left, right)
        else:
            print("Metric not recognized")
            exit()

    return sum / images.shape[0]


# All similarity measures:
# https://scikit-image.org/docs/dev/api/skimage.measure.html
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder", help="Folder where the images are")
    args = parser.parse_args()

    imgs_folder = args.folder
    if (len(sys.argv) == 1):
        imgs_folder = __TEST_FOLDER__

    images = np.array([cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in glob.glob(imgs_folder+"*.jpg")])

    ssim_avg = round(get_avg_metric(images, "ssim"), 3)
    psnr_avg = round(get_avg_metric(images, "psnr"), 3)
    mse_avg = round(get_avg_metric(images, "mse"), 3)

    print("\nFolder: {} | # of imgs: {}\n".format(imgs_folder, images.shape[0]))
    print("======== Averages ======== ")
    print("SSIM: {} \nPSNR: {} \nMSE: {}".format(ssim_avg, psnr_avg, mse_avg))
    print("========================== ")



