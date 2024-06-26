import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import json
import PIL
import random


def Softmax(x):
    x = np.exp(x)
    x = x / np.sum(x)
    return x


def derivative_tanh(x):
    return (1 - np.power(np.tanh(x), 2))


def derivative_relu(x):
    return np.array(x > 0, dtype=np.float32)

# Softmax(np.array([7, 0.5, 1, 3]))


class Network_training():
    def __init__(self, images, labels, epochs=5, learn_rate=0.03, lambd=0.02):
        self.images = images
        self.labels = labels
        self.lambd = lambd

        self.epochs = epochs
        self.learn_rate = learn_rate
        self.nr_correct = 0

        # 2 hidden layers
        self.first_w = np.random.uniform(-0.5, 0.5, (20, 784))
        self.first_b = np.zeros((20, 1))
        self.sec_w = np.random.uniform(-0.5, 0.5, (40, 20))
        self.sec_b = np.zeros((40, 1))

        # output layer
        self.output_w = np.random.uniform(-0.5, 0.5, (10, 40))
        self.output_b = np.zeros((10, 1))

    def random_test(self, number=4):
        image_list = np.array([])
        label_list = np.array([])
        predict_list = np.array([])

        for i in range(number):
            idx = random.randint(0, self.images.shape[0] - 1)
            image_list = np.append(image_list, self.images[idx])
            label_list = np.append(label_list, np.argmax(self.labels[idx]))
        image_list = image_list.reshape((number, -1))

        if number % 2 != 0:
            number += 1

        for i in range(2, 10):
            if number % i == 0:
                half = i

        f, axarr = plt.subplots(number // half, half)
        f.subplots_adjust(hspace=2, wspace=2)

        idx, x, y = 0, 0, 0
        for image in image_list:
            image = np.reshape(image, (image.shape[0], 1))

            # Forward Propogation
            pre_first = self.first_w @ image + self.first_b
            first = np.maximum(0, pre_first)

            pre_sec = self.sec_w @ first + self.sec_b
            sec = np.maximum(0, pre_sec)

            pre_out = self.output_w @ sec + self.output_b
            out = Softmax(pre_out)
            predict_list = np.append(predict_list, np.argmax(out))

            if idx % half == 0:
                y += 1
                x = 0
            text = 'Label: ' + str(label_list[idx]) + '\nModel: ' + str(predict_list[idx])
            axarr[y - 1, x].imshow(image.reshape(28, 28))
            axarr[y - 1, x].set_title(text)
            x += 1
            idx += 1

        plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
        plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
        plt.show()

    def load_model(self, path):
        file = np.load(path)
        self.nr_correct = file['nr_correct']
        self.output_w = file['output_w']
        self.output_b = file['output_b']
        self.first_w = file['first_w']
        self.first_b = file['first_b']
        self.sec_w = file['sec_w']
        self.sec_b = file['sec_b']
        # print(file.files)

    def run(self, isave=False, filename=None):
        result = np.zeros((10, 1))
        costs = np.array([])

        for epoch in range(self.epochs):
            correct = 0
            cost = 0
            for (image, label) in zip(self.images, self.labels):
                image = np.reshape(image, (image.shape[0], 1))
                label = np.reshape(label, (label.shape[0], 1))

                # Forward Propogation
                pre_first = self.first_w @ image + self.first_b
                first = np.maximum(0, pre_first)

                pre_sec = self.sec_w @ first + self.sec_b
                sec = np.maximum(0, pre_sec)

                pre_out = self.output_w @ sec + self.output_b
                out = Softmax(pre_out)
                result = out

                # Cost / Error fuction
                m = image.shape[0]
                L2 = (self.lambd / (2 * m)) * (np.sum(np.square(self.output_w)) + np.sum(np.square(self.first_w)) + np.sum(np.square(self.sec_w)))
                cost = -(1 / len(label)) * np.sum(label * np.log(out)) + L2
                correct += int(np.argmax(out) == np.argmax(label))

                # Backward Propoagation
                delta_o_Z = out - label
                delta_o_W = (1 / m) * delta_o_Z @ sec.T + self.lambd / m * self.output_w
                delta_o_b = (1 / m) * np.sum(delta_o_Z, axis=1, keepdims=True)
                self.output_w += - self.learn_rate * delta_o_W
                self.output_b += self.learn_rate * delta_o_b

                delta_sec_Z = self.output_w.T @ delta_o_Z * derivative_relu(sec)
                delta_sec_W = (1 / m) * delta_sec_Z @ first.T + self.lambd / m * self.sec_w
                delta_sec_b = (1 / m) * np.sum(delta_sec_Z, axis=1, keepdims=True)
                self.sec_w += -self.learn_rate * delta_sec_W
                self.sec_b += self.learn_rate * delta_sec_b

                delta_first_Z = self.sec_w.T @ delta_sec_Z * derivative_relu(first)
                delta_first_W = (1 / m) * delta_first_Z @ image.T + self.lambd / m * self.first_w
                delta_first_b = (1 / m) * np.sum(delta_first_Z, axis=1, keepdims=True)
                self.first_w += -self.learn_rate * delta_first_W
                self.first_b += self.learn_rate * delta_first_b

            percentage = np.round(correct / self.images.shape[0], 2) * 100
            print('epoch: ', epoch, '==> Cost', cost, ', Correctness', percentage)
            costs = np.append(costs, cost)

            if epoch == self.epochs - 1:
                if isave:
                    # Dump into model
                    if filename == None:
                        filename = str(int(percentage))
                    np.savez('models/' + filename, nr_correct=percentage, output_w=self.output_w, output_b=self.output_b, first_w=self.first_w, first_b=self.first_b, sec_w=self.sec_w, sec_b=self.sec_b)

                t = np.arange(0, len(costs))
                plt.plot(t, costs)
                plt.xlabel("Itterations")
                plt.ylabel("Cost")
                plt.show()
                # self.plot_values(self.images, self.labels, outs)

        return result


class Network_running():
    def __init__(self):
        self.model_url = None
        self.nr_correct = 0

        # 2 hidden layers
        self.first_w = np.random.uniform(-0.5, 0.5, (20, 784))
        self.first_b = np.zeros((20, 1))
        self.sec_w = np.random.uniform(-0.5, 0.5, (30, 20))
        self.sec_b = np.zeros((30, 1))

        # output layer
        self.output_w = np.random.uniform(-0.5, 0.5, (10, 30))
        self.output_b = np.zeros((10, 1))

        # Image for running
        self.image_run = None
        self.has_image = False

    def load_model(self, path):
        file = np.load(path)
        self.model_url = path
        self.nr_correct = file['nr_correct']
        self.output_w = file['output_w']
        self.output_b = file['output_b']
        self.first_w = file['first_w']
        self.first_b = file['first_b']
        self.sec_w = file['sec_w']
        self.sec_b = file['sec_b']
        # print(file.files)

    def load_image(self, path, showimg= False):
        self.image_run = np.array(PIL.Image.open(path))
        self.image_run = self.image_run.mean(axis=2)
        self.image_run = self.image_run.astype('float32') / 255
        if showimg:
            plt.imshow(self.image_run)
            plt.show()
        self.image_run = self.image_run.reshape((784, 1))
        self.has_image = True

    def load_image_np(self, np_matrix, showimg = False):
        self.image_run = np_matrix
        self.image_run = self.image_run.astype('float32') / 255
        if showimg:
            plt.imshow(self.image_run)
            plt.show()
        self.image_run = self.image_run.reshape((784, 1))
        self.has_image = True

    def run(self, showimg = False):
        result = np.zeros((10, 1))

        # Forward Propogation
        pre_first = self.first_w @ self.image_run + self.first_b
        first = np.maximum(0, pre_first)

        pre_sec = self.sec_w @ first + self.sec_b
        sec = np.maximum(0, pre_sec)

        pre_out = self.output_w @ sec + self.output_b
        out = Softmax(pre_out)
        result = out

        # # Cost / Error fuction
        # m = self.image_run.shape[0]
        # L2 = (self.lambd / (2 * m)) * (np.sum(np.square(self.output_w)) + np.sum(np.square(self.first_w)) + np.sum(np.square(self.sec_w)))
        # cost = -(1 / len(label)) * np.sum(label * np.log(out)) + L2

        # # Backward Propoagation
        # delta_o_Z = out - label
        # delta_o_W = (1 / m) * delta_o_Z @ sec.T + self.lambd / m * self.output_w
        # delta_o_b = (1 / m) * np.sum(delta_o_Z, axis=1, keepdims=True)
        # self.output_w += - self.learn_rate * delta_o_W
        # self.output_b += self.learn_rate * delta_o_b

        # delta_sec_Z = self.output_w.T @ delta_o_Z * derivative_relu(sec)
        # delta_sec_W = (1 / m) * delta_sec_Z @ first.T + self.lambd / m * self.sec_w
        # delta_sec_b = (1 / m) * np.sum(delta_sec_Z, axis=1, keepdims=True)
        # self.sec_w += -self.learn_rate * delta_sec_W
        # self.sec_b += self.learn_rate * delta_sec_b

        # delta_first_Z = self.sec_w.T @ delta_sec_Z * derivative_relu(first)
        # delta_first_W = (1 / m) * delta_first_Z @ image.T + self.lambd / m * self.first_w
        # delta_first_b = (1 / m) * np.sum(delta_first_Z, axis=1, keepdims=True)
        # self.first_w += -self.learn_rate * delta_first_W
        # self.first_b += self.learn_rate * delta_first_b

        if showimg:
            plt.imshow(self.image_run.reshape((28, 28)))
            plt.title('Model: ' + str(np.argmax(out)))
            plt.show()

        return result
