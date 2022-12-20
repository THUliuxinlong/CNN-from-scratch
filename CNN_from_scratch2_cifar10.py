import numpy as np
import random
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_data():
    batchsize = 128
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    # transform_train = transforms.Compose([
    #     # 在高度和宽度上将图像放大到40像素的正方形
    #     transforms.Resize(40),
    #     # 随机裁剪出一个高度和宽度均为40像素的正方形图像，
    #     # 生成一个面积为原始图像面积0.64到1倍的小正方形，
    #     # 然后将其缩放为高度和宽度均为32像素的正方形
    #     transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     # 标准化图像的每个通道
    #     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    #
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    cifar10_train = datasets.CIFAR10('data', True, transform=transform, download=True)
    trainData = DataLoader(cifar10_train, batch_size=batchsize, shuffle=True)

    cifar10_test = datasets.CIFAR10('data', False, transform=transform, download=True)
    testData = DataLoader(cifar10_test, batch_size=batchsize, shuffle=True)

    train_labels = []
    step = 0
    for imgs, labels in trainData:
        imgs = np.squeeze(imgs.numpy())
        labels = labels.numpy()

        if step == 0:
            train_data = imgs
            step += 1
        else:
            train_data = np.append(train_data, imgs, axis=0)
        train_labels = np.append(train_labels, labels, axis=0)
    train_labels = train_labels.astype(np.int32)
    print('traindata load ok!')
    print(train_data.shape, train_labels.shape)

    test_labels = []
    step = 0
    for imgs, labels in testData:
        imgs = np.squeeze(imgs.numpy())
        labels = labels.numpy()

        if step == 0:
            test_data = imgs
            step += 1
        else:
            test_data = np.append(test_data, imgs, axis=0)
        test_labels = np.append(test_labels, labels, axis=0)
    test_labels = test_labels.astype(np.int32)
    print('testdata load ok!')
    print(test_data.shape, test_labels.shape)

    return train_data, train_labels, test_data, test_labels


def conv_3d(array, kernel, b, stride=1):
    n, h, w = array.shape
    n_1, h_1, w_1 = kernel.shape
    new_image = np.zeros((h - h_1 + 1, w - w_1 + 1))
    delta = np.zeros(kernel.shape)
    for i in range(0, h - h_1 + 1):
        for j in range(0, w - w_1 + 1):
            new_image[i][j] = np.sum(array[:, i:i + h_1, j:j + w_1] * kernel) + b
    return new_image


class Conv(object):
    def __init__(self, kernel_shape, stride=1):
        n_out, n_in, wk, hk = kernel_shape

        self.stride = stride

        scale = np.sqrt(3 * wk * hk * n_in / n_out)
        self.k = np.random.standard_normal(kernel_shape) / scale
        self.b = np.random.standard_normal(n_out) / scale

        self.k_grad = np.zeros(kernel_shape)
        self.b_grad = np.zeros(n_out)

    def forward(self, x):
        self.x = x
        shape0 = self.x.shape
        shape1 = self.k.shape
        out = np.zeros((shape0[0], shape1[0], shape0[2] - shape1[2] + 1, shape0[3] - shape1[3] + 1))

        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i][j] = conv_3d(self.x[i], self.k[j], self.b[j])

        return out

    def bp(self, delta, lr):
        shape = delta.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                for n in range(shape[2]):
                    for m in range(shape[3]):
                        self.k_grad[j] += delta[i, j, n, m] * self.x[i, :, n:n + self.k.shape[2],
                                                              m:m + self.k.shape[3]]
        self.b_grad = np.sum(delta, axis=(0, 2, 3))
        self.k_grad /= shape[0]
        self.b_grad /= shape[0]

        '''计算x的梯度'''
        k_180 = np.rot90(self.k, 2, (2, 3))
        new_delta = np.zeros(self.x.shape)
        shape1 = self.x.shape
        padding = np.zeros(
            (shape1[0], shape[1], self.x.shape[2] + self.k.shape[2] - 1, self.x.shape[3] + self.k.shape[3] - 1))
        pad = (self.x.shape[2] + self.k.shape[2] - 1 - delta.shape[2]) // 2
        for i in range(padding.shape[0]):
            for j in range(padding.shape[1]):
                padding[i][j] = np.pad(delta[i][j], ((pad, pad), (pad, pad)), 'constant')
        k_180 = k_180.swapaxes(0, 1)

        shape0 = padding.shape
        shape1 = k_180.shape
        out = np.zeros((shape0[0], shape1[0], shape0[2] - shape1[2] + 1, shape0[3] - shape1[3] + 1))

        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i][j] = conv_3d(padding[i], k_180[j], 0)
        self.k -= lr * self.k_grad
        self.b -= lr * self.b_grad
        return out


def max_pooling(array):
    n, m = array.shape
    new_image = np.zeros((int(n / 2), int(m / 2)))
    delta_pooling = np.zeros((n, m))
    for i in range(0, int(n / 2)):
        for j in range(0, int(m / 2)):
            new_image[i][j] = np.max(array[i * 2:i * 2 + 2, j * 2:j * 2 + 2])
            index = np.unravel_index(array[i * 2:i * 2 + 2, j * 2:j * 2 + 2].argmax(),
                                     array[i * 2:i * 2 + 2, j * 2:j * 2 + 2].shape)
            middle = np.zeros((2, 2))
            middle[index[0]][index[1]] = 1
            delta_pooling[i * 2:i * 2 + 2, j * 2:j * 2 + 2] = middle
    return new_image, delta_pooling


class Pooling(object):
    def forward(self, x):
        self.x = x
        shape = self.x.shape
        out = np.zeros((shape[0], shape[1], shape[2] // 2, shape[3] // 2))
        self.delta = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                out[i][j], self.delta[i][j] = max_pooling(self.x[i][j])
        return out

    def bp(self, delta):
        shape = self.delta.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                for n in range(shape[2]):
                    for m in range(shape[3]):
                        if self.delta[i][j][n][m] == 1:
                            self.delta[i][j][n][m] = delta[i][j][n // 2][m // 2]
        return self.delta

    pass


class Linear(object):
    def __init__(self, input_size, output_size):
        scale = np.sqrt(input_size / 2)

        self.W = np.random.standard_normal((input_size, output_size)) / scale
        self.b = np.random.standard_normal(output_size) / scale

        self.W_grad = np.zeros((input_size, output_size))
        self.b_grad = np.zeros(output_size)

    def forward(self, x):
        self.x = x
        # reshape保证矩阵乘法的维度
        x = x.reshape((-1, self.W.shape[0]))
        out = np.dot(x, self.W) + self.b
        return out

    def bp(self, delta, lr):
        '''简单的反向传播过程'''
        shape = delta.shape
        self.b_grad = np.sum(delta, axis=0) / shape[0]
        # reshape保证矩阵乘法的维度
        self.x = self.x.reshape((delta.shape[0], -1))
        self.W_grad = np.dot(self.x.T, delta) / shape[0]
        # # reshape保证矩阵乘法的维度
        # delta = delta.reshape(-1, self.W.T[0])
        new_delta = np.dot(delta, self.W.T)

        self.W -= lr * self.W_grad
        self.b -= lr * self.b_grad

        return new_delta


class Relu(object):
    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, delta):
        delta[self.x < 0] = 0
        return delta


class Softmax(object):
    def forward(self, x, y):
        self.x = x
        shape = self.x.shape
        out = np.exp(self.x - np.max(self.x))
        for i in range(shape[0]):
            sums = np.sum(out[i, :])
            for j in range(shape[1]):
                out[i][j] = out[i][j] / sums
        loss = 0
        delta = np.zeros(shape)
        for i in range(shape[0]):
            delta[i] = out[i] - y[i]
            for j in range(shape[1]):
                loss += - y[i][j] * np.log(out[i][j])
        loss /= shape[0]
        return loss, delta

    pass


def get_batchsize(batch_size, N):
    a = []
    b = list(range(N))
    random.shuffle(b)
    for i in range(N):
        l = b[i * batch_size:batch_size * (i + 1)]
        a.append(l)
        if len(l) < batch_size:
            break
    return a


def train():
    # 随机选6000个作为训练集
    N = 20000
    random_index = random.sample(range(train_data.shape[0]), N)
    train_x = train_data[random_index]
    train_y = train_labels[random_index]
    # train_x = train_data
    # train_y = train_labels
    N = len(train_y)

    oneHot = np.identity(10)
    train_y = oneHot[train_y]
    # train_x = train_x.reshape(len(train_y), 1, 28, 28) / 255
    train_x = train_x.reshape(len(train_y), 3, 32, 32)

    conv1 = Conv(kernel_shape=(6, 3, 5, 5))  # N * 6 * 24 * 24
    relu1 = Relu()
    pool1 = Pooling()  # N * 6 * 12 *12

    conv2 = Conv(kernel_shape=(16, 6, 5, 5))  # N * 16 * 8 * 8
    relu2 = Relu()
    pool2 = Pooling()  # N * 16 * 4 * 4

    linear1 = Linear(400, 120)
    linear2 = Linear(120, 84)
    linear3 = Linear(84, 10)

    softmax = Softmax()
    epoch = 5
    batch_size = 64
    lr = 0.01
    for i in range(epoch):
        batch_radom_index = get_batchsize(batch_size, N)
        for n, indexs in enumerate(batch_radom_index):
            if len(indexs) == 0:
                break
            batch_x = train_x[indexs]
            batch_y = train_y[indexs]
            # # myimg = batch_x[0] * 255
            # myimg = batch_x[0]
            # myimg = myimg.swapaxes(0, 1)
            # myimg = myimg.swapaxes(1, 2)
            # plt.imshow(myimg)  # 绘制图片
            # plt.show()
            # print(batch_x[0][1][6])
            # print(batch_y[0])
            # print(batch_x.shape, batch_y.shape)
            out = conv1.forward(batch_x)
            # print('out.shape', out.shape)
            out = relu1.forward(out)
            # print('out.shape', out.shape)
            out = pool1.forward(out)
            # print('out.shape', out.shape)

            out = conv2.forward(out)
            # print('out.shape', out.shape)
            out = relu2.forward(out)
            # print('out.shape', out.shape)
            out = pool2.forward(out)
            # print('out.shape', out.shape)

            out = out.reshape(batch_size, -1)
            # print('out.shape', out.shape)

            out = linear1.forward(out)
            # print('out.shape', out.shape)
            out = linear2.forward(out)
            # print('out.shape', out.shape)
            out = linear3.forward(out)
            # print('out.shape', out.shape)

            loss, delta = softmax.forward(out, batch_y)
            # print('delta.shape', delta.shape)

            delta = linear3.bp(delta, lr)
            # print('delta.shape', delta.shape)
            delta = linear2.bp(delta, lr)
            # print('delta.shape', delta.shape)
            delta = linear1.bp(delta, lr)
            # print('delta.shape', delta.shape)
            delta = np.resize(delta, (batch_size, 400))
            # print('delta.shape', delta.shape)
            delta = delta.reshape((batch_size, 16, 5, 5))
            # print('delta.shape', delta.shape)

            delta = pool2.bp(delta)
            delta = relu2.backward(delta)
            delta = conv2.bp(delta, lr)

            delta = pool1.bp(delta)
            delta = relu1.backward(delta)
            conv1.bp(delta, lr)

            print("Epoch-{}-{:05d}".format(str(i), n), ":", "loss:{:.4f}".format(loss))
        lr *= 0.95 ** (i + 1)  # 学习率指数衰减
        savepath = 'model_' + str(i) + '.npz'
        np.savez(savepath, k1=conv1.k, b1=conv1.b,
                 k2=conv2.k, b2=conv2.b,
                 w3=linear1.W, b3=linear1.b,
                 w4=linear2.W, b4=linear2.b,
                 w5=linear3.W, b5=linear3.b)


def test():
    r = np.load("model_4.npz")  # 载入训练好的参数
    # 随机选1000个作为测试集
    N = 3000
    random_index = random.sample(range(test_data.shape[0]), N)
    test_x = test_data[random_index]
    test_y = test_labels[random_index]
    # test_x = test_data
    # test_y = test_labels
    N = len(test_x)
    # oneHot = np.identity(10)
    # test_y = oneHot[test_y]
    test_x = test_x.reshape(N, 3, 32, 32)  # 归一化

    conv1 = Conv(kernel_shape=(6, 3, 5, 5))  # N * 6 * 24 * 24
    relu1 = Relu()
    pool1 = Pooling()  # N * 6 * 12 *12

    conv2 = Conv(kernel_shape=(16, 6, 5, 5))  # N * 16 * 8 * 8
    relu2 = Relu()
    pool2 = Pooling()  # N * 16 * 4 * 4

    nn1 = Linear(400, 120)
    nn2 = Linear(120, 84)
    nn3 = Linear(84, 10)

    softmax = Softmax()

    conv1.k = r["k1"]
    conv1.b = r["b1"]
    conv2.k = r["k2"]
    conv2.b = r["b2"]
    nn1.W = r["w3"]
    nn1.b = r["b3"]
    nn2.W = r["w4"]
    nn2.b = r["b4"]
    nn3.W = r["w5"]
    nn3.b = r["b5"]

    out = conv1.forward(test_x)
    out = relu1.forward(out)
    out = pool1.forward(out)
    out = conv2.forward(out)
    out = relu2.forward(out)
    out = pool2.forward(out)

    out = out.reshape(N, -1)

    out = nn1.forward(out)
    out = nn2.forward(out)
    out = nn3.forward(out)

    num = 0
    for i in range(N):
        if np.argmax(out[i, :]) == test_y[i]:
            num += 1
    print("TEST-ACC: ", num / N * 100, "%")


if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = load_data()
    train()
    test()
