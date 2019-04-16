import os
import cv2
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import imageio

from scipy import misc

__TEST_IMG__ = "data/dog.jpg"
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

    x2, y2 = np.random.randint(50, min(x1 - offset, y1 - offset),
                               size=2)  # force second rectangle in top left corner above first rectangle
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

def make_directory(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:  # results directory path already exists
        pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Image defects dataset creator')
    parser.add_argument('--rootdir', type=str, help='Input dir for images')
    parser.add_argument('--i', default=15, type=int, help='Intensity of the blur kernel')
    parser.add_argument('--defect', default="blur", type=str, help='Image defect: blur or color')
    parser.add_argument('--random', default=True, type=bool, help='Randomize patches')
    parser.add_argument('--split_percent', default=50, type=int, help='Split percentage of train vs test images')
    parser.add_argument('--max_images', default=500, type=int, help='Max number of images to process in the dataset')
    parser.add_argument('--resize_crop', default="crop", type=str, help='Choose to resize or crop the images')
    parser.add_argument('--res_folder', default="results/", type=str, help='Choose to resize or crop the images')

    args = parser.parse_args()

    root_name = args.rootdir.split("/")[0]
    PATH = args.res_folder+root_name+"/"+root_name

    if args.random:
        PATH += "_random"
    if args.defect == "blur":
        PATH += "_blur_" + str(args.i)
    if args.defect == "color":
        PATH += "_color"


    save_train_dir = PATH+"/train"
    save_test_dir = PATH+"/test"
    originals_save_train_dir = PATH + "_ORIGINALS/train"
    originals_save_test_dir = PATH + "_ORIGINALS/test"

    make_directory(save_train_dir)
    make_directory(save_test_dir)

    make_directory(originals_save_train_dir)
    make_directory(originals_save_test_dir)

    total_imgs = len([name for name in os.listdir(root_name) if os.path.isfile(os.path.join(root_name, name))])

    train_split = round(total_imgs * (args.split_percent/100))
    test_split = total_imgs - train_split

    curr_save_dir = save_train_dir
    curr_originals_save_dir = originals_save_train_dir
    for i, file in enumerate(glob.glob(args.rootdir + "*.*")):
        name, ext = os.path.splitext(file.split("/")[-1])
        img = get_img(file)

        if (args.resize_crop == "crop"):
            img = img[30:286, 100:356].copy()  # crop image
        else:
            img = cv2.resize(img, (256, 256))  # resize image

        mask_zeros = np.zeros_like(img)

        if (args.defect == "blur"):
            aux = apply_blur(img, intensity=args.i)
        else:
            aux = np.full_like(img, 255)  # white = 255, gray = 128, black = 0
            # black_img = np.full_like(img, 0)
            # aux = np.zeros((256, 256, 3), 'uint8')
            # aux[..., 0] = 0
            # aux[..., 1] = 0
            # aux[..., 2] = 0

        if args.random:
            mask_zeros = get_random_rectangle_mask(mask_zeros)
        else:
            mask_zeros = get_rectangle_mask(mask_zeros)
            # mask_zeros = get_circle_mask(mask_zeros)

        img_with_blurs = np.where(mask_zeros == np.array([255, 255, 255]), aux, img)
        imgs_side_2_side = np.hstack((img, img_with_blurs))

        # Save images
        imageio.imsave(curr_save_dir+"/"+str(i+1)+ext, imgs_side_2_side)
        imageio.imsave(curr_originals_save_dir+"/"+str(i+1)+ext, img)

        if i == args.max_images:
            break
        elif i == train_split:
            curr_save_dir = save_test_dir
            curr_originals_save_dir = originals_save_test_dir
