import numpy as np
import cv2
import matplotlib.pyplot as plt



def display_img(rgb_img):
    plt.imshow(rgb_img)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

def get_img():
    bgr_img = cv2.imread("data/dog.jpg")
    b, g, r = cv2.split(bgr_img)  # get b,g,r
    rgb_img = cv2.merge([r, g, b])  # switch it to rgb

    return rgb_img

if __name__ == '__main__':
    img = get_img()
    mask = np.zeros_like(img)

    cv2.rectangle(mask, (0, 0), (100, 100), (255,255,255), -1)
    masked_img_1 = cv2.bitwise_and(img, mask)

    print(masked_img_1)
    display_img(masked_img_1)


    # masked_img = cv2.bitwise_and(img, mask)




