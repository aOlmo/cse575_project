import os
import sys
import cv2
import glob
import imageio
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

def get_avg_metric(images, metric, split, save_dir_root):
    height, width = images.shape[1], images.shape[2]

    width_cutoff = width // 2
    s1 = images[..., :width_cutoff, :]
    s2 = images[..., width_cutoff:, :]

    sum = 0
    i = 0
    for left, right in zip(s1, s2):
        if split:
            imageio.imsave(save_dir_root+"/sharp/" + str(i) + ".png", left)
            imageio.imsave(save_dir_root+"/blur/" + str(i) + ".png", right)

        if (metric == "ssim"):
            sum += ssim(left, right, data_range=right.max() - right.min(), multichannel=True)
        elif (metric == "psnr"):
            sum += psnr(left, right)
        elif (metric == "mse"):
            sum += mse(left, right)
        else:
            print("Metric not recognized")
            exit()
        i += 1

    return sum / images.shape[0]

def make_directory(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:  # results directory path already exists
        pass

# All similarity measures:
# https://scikit-image.org/docs/dev/api/skimage.measure.html
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder", help="Folder where the images are")
    parser.add_argument("--split_sharp_blur", type=bool, help="Splits the images into two new folders")

    args = parser.parse_args()
    split = args.split_sharp_blur
    imgs_folder = args.folder

    if (args.folder.split("/")[-1] != ""):
        root_name = args.folder.split("/")[-1]
    else:
        root_name = args.folder.split("/")[-2]

    if split:
        save_dir_root = "results/"+root_name+"/"
        make_directory(save_dir_root+"/sharp")
        make_directory(save_dir_root+"/blur")

    images = np.array([cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in glob.glob(imgs_folder+"*.png")])

    ssim_avg = round(get_avg_metric(images, "ssim", split, save_dir_root), 3)
    psnr_avg = round(get_avg_metric(images, "psnr", split, save_dir_root), 3)
    mse_avg = round(get_avg_metric(images, "mse", split, save_dir_root), 3)

    print("\nFolder: {} | # of imgs: {}\n".format(imgs_folder, images.shape[0]))
    print("======== Averages ======== ")
    print("SSIM: {} \nPSNR: {} \nMSE: {}".format(ssim_avg, psnr_avg, mse_avg))
    print("========================== ")



