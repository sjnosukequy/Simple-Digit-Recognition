import cv2 as cv
import numpy as np
import random
import PIL
import matplotlib.pyplot as plt


def Rotate(image, angle):
    height, width = image.shape[:2]
    rotation_matrix = cv.getRotationMatrix2D((height / 2, width / 2), angle, 1)
    rotated_image = cv.warpAffine(image, rotation_matrix, (height, width))
    return rotated_image


def Scale(image, scaler):
    height, width = image.shape[:2]
    fin_h, fin_w = int(height * scaler), int(width * scaler)
    cen_x, cen_y = width // 2, height // 2
    crop_img = np.zeros((height, width))

    temp = cv.resize(image, (fin_h, fin_w), interpolation=cv.INTER_LINEAR)
    temp_cen_x, temp_cen_y = fin_w // 2, fin_h // 2

    min_x, min_y = min(width, fin_w), min(height, fin_h)
    for i in range(min_y):
        for k in range(min_x):
            if scaler > 1:
                # print(k, temp_cen_x + (k - (min_x) // 2))
                crop_img[cen_y + (i - cen_y)][cen_x + (k - cen_x)] = temp[temp_cen_y + (i - (min_y) // 2)][temp_cen_x + (k - (min_x) // 2)]
            elif scaler < 1:
                crop_img[cen_y + (i - min_y // 2)][cen_x + (k - min_x // 2)] = temp[temp_cen_y + (i - temp_cen_y)][temp_cen_x + (k - temp_cen_x)]
            else:
                crop_img[cen_y + (i - cen_y)][cen_x + (k - cen_x)] = temp[temp_cen_y + (i - temp_cen_y)][temp_cen_x + (k - temp_cen_x)]
    return crop_img


def Offset(image, x, y):
    # Specify the X and Y offset values
    offset_x = x  # pixels
    offset_y = y  # pixels

    # Define the transformation matrix M
    M = np.float32([[1, 0, offset_x],   # 1st row: [1, 0, offset_x]
                    [0, 1, offset_y]])  # 2nd row: [0, 1, offset_y]

    # Apply the affine transformation using warpAffine
    offset_image = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return offset_image


def random_img(image, reshape=False):
    scale_int = random.randint(8, 12) / 10
    angle = random.randint(-10, 10)
    img = Scale(image, scale_int)
    height, width = img.shape[:2]
    offset_x, offset_y = random.randint(-2, 2), random.randint(-2, 2)
    img = Rotate(img, angle)
    img = Offset(img, offset_x, offset_y)
    # img = img.astype('float32') / 255

    if reshape:
        img = img.reshape(784, 1)
        return img
    else:
        return img


def load_img(path, showimg=False):
    '''
    Return image with shape (784, 1). This is for the model.
    you need to reshape it back to (28, 28) if you want to use it for anything else.
    '''
    image = np.array(PIL.Image.open(path))
    image = cv.resize(image, (28, 28))
    image = image.mean(axis=2)
    image = image.astype('float32') / 255
    if showimg:
        plt.imshow(image)
        plt.show()
    image = image.reshape((784, 1))
    return image


def loading_mnist(regular = True):
    files = np.load('data/mnist.npz')
    # print(files.files) # ['x_test', 'x_train', 'y_train', 'y_test']
    # x_train for images and y_train for tags
    images, labels = files['x_train'], files['y_train']
    # print(images.shape)
    # print(labels.shape)
    # plt.imshow(images[0])
    # plt.show()
    if regular:
        images = images.astype('float32') / 255
    # plt.imshow(images[0])
    # plt.show()
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]
    # images shape is (60000, 784)
    # labeles shape is (60000, 10)
    return images, labels

def loading_random(path, regular = True):
    files = np.load(path)
    # print(files.files) # ['x_test', 'x_train', 'y_train', 'y_test']
    # x_train for images and y_train for tags
    images, labels = files['x_train'], files['y_train']
    if regular:
        images = images.astype('float32') / 255
    return images, labels

def create_random_dataset(images, labels, filename):
    images = np.array([random_img(image.reshape((28, 28)), True) for image in images])
    # plt.imshow(images[2].reshape((28, 28)))
    # plt.show()
    np.savez_compressed('data/' + filename, x_train=images, y_train=labels)
    print('Created:', 'data/' + filename +'.npz')