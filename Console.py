import numpy as np
import cv2 as cv
from Network import Network_training, Network_running
import matplotlib.pyplot as plt
from Utils import *


if __name__ == '__main__':

    ## MERGING DATASETS
    # images1, labels1 = loading_random('data/random#lite2.npz', regular=False)
    # # images2, labels2 = loading_random('data/random#2.npz', regular=False)
    # # images3, labels3 = loading_random('data/random#3.npz', regular=False)
    # images, labels = loading_mnist(regular=False)

    # img = np.append(images, images1)
    # # img = np.append(img, images2)
    # # img = np.append(img, images3)
    # lb = np.append(labels, labels1)
    # # lb = np.append(lb, labels2)
    # # lb = np.append(lb, labels3)

    # img = img.reshape((-1, 784))
    # lb = lb.reshape((-1, 10))
    # np.savez_compressed('data/' + 'ultima_lite', x_train=img, y_train=lb)
    # print(img.shape)
    
    ## COMPARING DATASETS
    # images, labels = loading_mnist()
    # images1, labels1 = loading_random('data/random#lite2.npz')
    # rand = random.randint(0, 6000)
    # f, fig = plt.subplots(1, 2)
    # fig[0].imshow(images1[rand].reshape((28, 28)))
    # fig[0].set_title(np.argmax(labels1[rand]))
    # fig[1].imshow(images[rand].reshape((28, 28)))
    # fig[1].set_title(np.argmax(labels[rand]))
    # plt.show()

    ## CREATE RANDOM IMAGES FROM DATASET
    # images, labels = loading_mnist(regular=False)
    # rotated_image = random_img(images[0].reshape((28,28)))
    # create_random_dataset(images, labels, 'random#lite3')

    ## LOAD AN IMAGE
    # img = load_img('test.png', False)
    # img = img.reshape((28, 28))
    # img = random_img(img)
    # plt.imshow(img)
    # plt.show()
    
    ## TRAINING MODEL
    images, labels = loading_random('data/ultima_lite2.npz')
    network = Network_training(images=images, labels=labels, epochs=200, learn_rate=0.02, lambd=0.01)
    # network.load_model('models/90.npz') # IF YOU WANT TO TRAIN THE MODEL FROM THE START COMMENT OUT THIS LINE
    result = network.run(isave=True, filename='new')
    network.random_test(number=40)
    
    ## RUNNING MODEL
    # images, labels = loading_random('data/ultima_lite.npz')
    # network_run = Network_running()
    # network_run.load_model('models/90.npz')
    # network_run.load_image('test.png')
    # result = network_run.run(showimg=True)
    # stats = result * 100
    # stats = np.round(stats, 3)
    # print(stats)
    # print(np.argmax(result))
