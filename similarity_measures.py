import os
import sys
import cv2
import glob
from tqdm import tqdm
import imageio
import numpy as np
import argparse
import matplotlib.pyplot as plt

from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_mse as mse


__TEST_FOLDER__ = "results/test/"
MAX_IMAGES = 512

def display_img(rgb_img):
    plt.imshow(rgb_img)
    plt.xticks([]), plt.yticks([])
    plt.show()

def print_metrics(ssim_sum, psnr_sum, i):
    print("[+]: Iteration {}\nAvg SSIM: {}, Avg PSNR: {}".format(i, round(ssim_sum/i, 3), round(psnr_sum/i, 3)))


def get_avg_metrics(images, split, save_dir_root):
    height, width = images.shape[1], images.shape[2]

    width_cutoff = width // 2
    s1 = images[..., :width_cutoff, :]
    s2 = images[..., width_cutoff:, :]

    i = 0
    ssim_sum = 0
    psnr_sum = 0
    for left, right in tqdm(zip(s1, s2), total=MAX_IMAGES):
        i += 1

        if split:
            imageio.imsave(save_dir_root+"/sharp/" + str(i) + ".png", left)
            imageio.imsave(save_dir_root+"/blur/" + str(i) + ".png", right)

        ssim_sum += ssim(left, right, data_range=right.max() - right.min(), multichannel=True)
        psnr_sum += psnr(left, right)

        if (i % 50 == 0) and i != 0:
            print("[+]: Iteration {}".format(i))
            print_metrics(ssim_sum, psnr_sum, i)

        if i == MAX_IMAGES:
            break

    return ssim_sum / i, psnr_sum / i

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

    parser.set_defaults(split_sharp_blur=True)

    args = parser.parse_args()

    split = args.split_sharp_blur
    imgs_folder = args.folder

    if (args.folder.split("/")[-1] != ""):
        root_name = args.folder.split("/")[-1]
    else:
        root_name = args.folder.split("/")[-2]

    save_dir_root = ""
    if split:
        save_dir_root = "results/"+root_name+"/"
        make_directory(save_dir_root+"/sharp")
        make_directory(save_dir_root+"/blur")

    images = np.array([cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in sorted(glob.glob(imgs_folder+"*.png"))])

    ssim_avg, psnr_avg = get_avg_metrics(images, split, save_dir_root)

    print("\nFolder: {} | # of imgs: {}\n".format(imgs_folder, images.shape[0]))
    print("======== Averages ======== ")
    print("SSIM: {} \nPSNR: {}".format(ssim_avg, psnr_avg))
    print("========================== ")



