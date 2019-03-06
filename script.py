import numpy as np
import cv2
import matplotlib.pyplot as plt

__TEST_IMG = "data/dog.jpg"

def display_img(rgb_img):
    plt.imshow(rgb_img)
    # to hide tick values on X and Y axis
    plt.xticks([]), plt.yticks([])
    plt.show()


def get_img():
    bgr_img = cv2.imread(__TEST_IMG)
    # get bgr and switch to rgb
    b, g, r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r, g, b])
    return rgb_img


if __name__ == '__main__':
    img = get_img()
    # When performing the bitwise and we need to have 0s in the mask
    # and the original RGB value for the other regions, thus in a
    # Bitwise operation, we need whatever RGB value is and 2)11111111 = 10)255
    mask = np.full_like(img, 255)

    # Drawing of our masks
    cv2.rectangle(mask, (265, 17), (380, 137), 0, -1)
    cv2.rectangle(mask, (0,280), (100, 320), 0, -1)

    print(mask)

    # Apply the mask and display the result
    maskedImg = cv2.bitwise_and(img, mask)
    display_img(maskedImg)

