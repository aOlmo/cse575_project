import os
import cv2
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import imageio

from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_mse as mse

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

def get_metrics(images, metric):
    height, width = images.shape[0], images.shape[1]

    width_cutoff = width // 2
    s1 = images[..., :width_cutoff, :]
    s2 = images[..., width_cutoff:, :]

    s1[s1==np.nan] = 0
    s2[s2==np.nan] = 0

    sum = 0
    s1 = np.expand_dims(s1, 0)
    s2 = np.expand_dims(s2, 0)

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

    return sum

def print_metrics(ssim_sum, psnr_sum, i):
    print("[+]: Iteration {}\nAvg SSIM: {}, Avg PSNR: {}".format(i, round(ssim_sum/i, 3), round(psnr_sum/i, 3)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Image defects dataset creator')
    parser.add_argument('--rootdir', type=str, help='Input dir for images')
    parser.add_argument('--i', default=15, type=int, help='Intensity of the blur kernel')
    parser.add_argument('--defect', default="blur", type=str, help='Image defect: blur, color or full_blur')
    parser.add_argument('--split_percent', default=100, type=float, help='Split percentage of train vs test images') # 79.72
    parser.add_argument('--max_images', type=int, help='Max number of images to process in the dataset')
    parser.add_argument('--resize_crop', default="resize", type=str, help='Choose to resize or crop the images')
    parser.add_argument('--res_folder', default="results/", type=str, help='Folder where the results will be saved')

    parser.add_argument('--save_as_pix2pix_format', action="store_true", help='Flag to save the resulting images in Pix2Pix format')
    parser.add_argument('--no_save_as_pix2pix_format', action="store_false")

    parser.add_argument('--save_originals', action="store_true", dest="save_originals", help='Flag to only save the image defects')
    parser.add_argument('--no_save_originals', action="store_false", dest="save_originals")

    parser.add_argument('--random', action="store_true", dest="random", help='Randomize patches')
    parser.add_argument('--no_random', action="store_false", dest="random")

    parser.set_defaults(random=True)
    parser.set_defaults(save_originals=False)
    parser.set_defaults(save_as_pix2pix_format=False)

    args = parser.parse_args()

    root_path = args.rootdir

    if (args.rootdir.split("/")[-1] != ""):
        root_name = args.rootdir.split("/")[-1]
    else:
        root_name = args.rootdir.split("/")[-2]

    PATH = args.res_folder + root_name + "/" + root_name

    if args.random and args.defect != "full_blur":
        PATH += "_random"
    if args.defect == "blur":
        PATH += "_blur_" + str(args.i)
    if args.defect == "color":
        PATH += "_color"
    if args.defect == "full_blur":
        PATH += "_full_blur_" + str(args.i)

    save_train_dir = PATH + "/train"
    save_test_dir = PATH + "/test"
    originals_save_train_dir = PATH + "_ORIGINALS/train"
    originals_save_test_dir = PATH + "_ORIGINALS/test"


    make_directory(save_train_dir)
    if args.split_percent < 100:
        make_directory(save_test_dir)

    if args.save_originals:
        make_directory(originals_save_train_dir)
        if args.split_percent < 100:
            make_directory(originals_save_test_dir)

    total_imgs = len([name for name in os.listdir(root_path) if os.path.isfile(os.path.join(root_path, name))])

    train_split = int(round(total_imgs * (args.split_percent / 100)))
    test_split = total_imgs - train_split

    curr_save_dir = save_train_dir
    curr_originals_save_dir = originals_save_train_dir

    print("# Training: {} \n# Testing: {}".format(train_split, test_split))


    ssim_sum = 0
    psnr_sum = 0
    for i, file in enumerate(glob.glob(args.rootdir + "/*.*")):
        name, ext = os.path.splitext(file.split("/")[-1])
        img = get_img(file)

        if args.resize_crop == "crop":
            img = img[30:286, 100:356].copy()  # crop image
        elif args.resize_crop == "resize":
            img = cv2.resize(img, (256, 256))  # resize image

        mask_zeros = np.zeros_like(img)

        if (args.defect == "blur" or args.defect == "full_blur"):
            aux = apply_blur(img, intensity=args.i)
        else:
            aux = np.full_like(img, 255)  # white = 255, gray = 128, black = 0
            # black_img = np.full_like(img, 0)
            # aux = np.zeros((256, 256, 3), 'uint8')
            # aux[..., 0] = 0
            # aux[..., 1] = 0
            # aux[..., 2] = 0

        if args.random and args.defect != "full_blur":
            mask_zeros = get_random_rectangle_mask(mask_zeros)
        else:
            mask_zeros = get_rectangle_mask(mask_zeros)

        if args.defect == "full_blur":
            mask_zeros = np.full_like(img, 255)

        img_with_blurs = np.where(mask_zeros == np.array([255, 255, 255]), aux, img)
        imgs_side_2_side = np.hstack((img, img_with_blurs))

        ssim_sum += get_metrics(imgs_side_2_side, "ssim")
        psnr_sum += get_metrics(imgs_side_2_side, "psnr")

        # Saving images
        # --------------------------------------------------------------------------
        if not args.save_as_pix2pix_format:
            imageio.imsave(curr_save_dir + "/" + str(i + 1) + ext, img_with_blurs)
        else:
            imageio.imsave(curr_save_dir + "/" + str(i + 1) + ext, imgs_side_2_side)

        if args.save_originals:
            imageio.imsave(curr_originals_save_dir + "/" + str(i + 1) + ext, img)
        # --------------------------------------------------------------------------

        if i == args.max_images:
            print("[+]: Breaking {}, {}".format(i, args.max_images))
            break
        elif i == train_split:
            curr_save_dir = save_test_dir
            curr_originals_save_dir = originals_save_test_dir

        if (i % 50 == 0) and i != 0:
            print("[+]: Iteration {}/{}".format(i, total_imgs))
            print_metrics(ssim_sum, psnr_sum, i)

    print("[+]: End at iteration {}\nAvg SSIM: {}, Avg PSNR: {}".format(i, round(ssim_sum/i, 4), round(psnr_sum/i, 4)))
