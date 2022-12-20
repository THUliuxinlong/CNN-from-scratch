import numpy as np


class Conv3x3:
    # 使用 3x3 filters 的卷积层

    def __init__(self, num_filters):
        self.num_filters = num_filters

        # filters(num_filters, 3, 3)
        # 除以9，对于初始化的值不能太大也不能太小
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        # image是一个2d numpy数组
        h, w = image.shape

        # 将 im_region, i, j 以 tuple 形式存储到迭代器中
        # 以便后面遍历使用
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    def forward(self, input):
        # input 为 image，即输入数据
        # output 为输出框架，默认都为 0，都为 1 也可以，反正后面会覆盖
        # input: 28x28
        # output: 26x26x8
        self.last_input = input

        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            # 卷积运算，点乘再相加，ouput[i, j] 为向量
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

        return output

    def backprop(self, d_L_d_out, learn_rate):
        # d_L_d_out: 这一层输出损失的梯度
        # learn_rate: a float
        # 初始化一组为 0 的 gradient，3x3x8
        d_L_d_filters = np.zeros(self.filters.shape)

        # im_region，一个个 3x3 小矩阵
        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                # 按 f 分层计算，一次算一层，然后累加起来
                # d_L_d_filters[f]: 3x3 matrix
                # d_L_d_out[i, j, f]: num
                # im_region: 3x3 matrix in image
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        # 更新filters
        self.filters -= learn_rate * d_L_d_filters

        # 这里我们不返回任何东西，因为我们使用Conv3x3作为CNN的第一层。否则，我们需要为这个层的输入返回损失梯度，就像CNN中的其他层一样。
        return None


class MaxPool2:
    # 最大池化 pool size 2.

    def iterate_regions(self, image):
        # image: 26x26x8
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def forward(self, input):
        # input(h, w, num_filters)
        # return(h / 2, w / 2, num_filters).
        self.last_input = input

        # input: 3d matrix of conv layer
        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))

        return output

    def backprop(self, d_L_d_out):
        # d_L_d_out: the loss gradient for the layer's outputs
        # 池化层输入数据，26x26x8，默认初始化为 0
        d_L_d_input = np.zeros(self.last_input.shape)

        # 每一个 im_region 都是一个 3x3x8 的8层小矩阵
        # 修改 max 的部分，首先查找 max
        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            # 获取 im_region 里面最大值的索引向量，一叠的感觉
            amax = np.amax(im_region, axis=(0, 1))

            # 遍历整个 im_region，对于传递下去的像素点，修改 gradient 为 loss 对 output 的gradient
            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # If this pixel was the max value, copy the gradient to it.
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i + i2, j + j2, f2] = d_L_d_out[i, j, f2]

        return d_L_d_input


class Softmax:
    # 标准全连接层用softmax激活

    def __init__(self, input_len, nodes):
        # We divide by input_len to reduce the variance of our initial values
        # input_len: length of input nodes
        # nodes: lenght of ouput nodes

        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        # 3d， 13x13x8
        self.last_input_shape = input.shape

        # 3d to 1d 用来构建全连接网络
        input = input.flatten()

        # 1d vector after flatting
        self.last_input = input

        input_len, nodes = self.weights.shape

        # input: 13x13x8 = 1352
        # self.weights: (1352, 10)
        # 以上叉乘之后为 向量，1352个节点与对应的权重相乘再加上bias得到输出的节点
        # totals: 向量, 10
        totals = np.dot(input, self.weights) + self.biases

        # softmax 前的向量，10
        self.last_totals = totals

        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)

    def backprop(self, d_L_d_out, learn_rate):
        # only 1 element of d_L_d_out is nonzero
        for i, gradient in enumerate(d_L_d_out):
            # k != c, gradient = 0
            # k == c, gradient = 1
            # try to find i when k == c
            # 找到 label 的值，就是 gradient 不为 0 的
            if gradient == 0:
                continue

            # e^totals
            t_exp = np.exp(self.last_totals)

            # Sum of all e^totals
            S = np.sum(t_exp)

            # Gradients of out[i] against totals
            # all gradients are given value with k != c
            # 初始化都设置为 非 c 的值，再单独修改 c 的值
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            # change the value of k == c
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            # Gradients of out[i] against totals
            # gradients to every weight in every node
            # this is not the final results
            # Gradients of totals against weights/biases/input
            # d_t_d_w 的结果是 softmax 层的输入数据，1352 个元素的向量
            # 不是最终的结果，最终结果是 2d 矩阵，1352x10
            d_t_d_w = self.last_input  # vector
            d_t_d_b = 1
            # 1000 x 10
            # d_t_d_input 的结果是 weights 值，2d 矩阵，1352x10
            d_t_d_inputs = self.weights

            # Gradients of loss against totals 向量，10
            # d_L_d_t, d_out_d_t, vector, 10 elements
            d_L_d_t = gradient * d_out_d_t

            # Gradients of loss against weights/biases/input
            # np.newaxis 可以帮助一维向量变成二维矩阵
            # (1352, 1) @ (1, 10) to (1352, 10)
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            # (1352, 10) @ (10, 1) to (1352, 1)
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            # Update weights / biases
            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b

            # it will be used in previous pooling layer
            # reshape into that matrix
            # 将矩阵从 1d 转为 3d
            # 1352 to 13x13x8
            return d_L_d_inputs.reshape(self.last_input_shape)


def cross_entropy_loss(out, label):
    # 交叉熵损失函数
    label = np.argwhere(label == 1)
    label = int(label)
    loss = -np.log(out[label])
    return loss


def selfCNN(image, label):
    # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
    # to work with. This is standard practice.

    # out 为卷基层的输出, 26x26x8
    out = conv.forward((image / 255) - 0.5)
    # out 为池化层的输出, 13x13x8
    out = pool.forward(out)
    # out 为 softmax 的输出, 10
    out = softmax.forward(out)

    # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
    # 损失函数的计算只与 label 的数有关，相当于索引
    loss = cross_entropy_loss(out, label)
    # 如果 softmax 输出的最大值就是 label 的值，表示正确，否则错误
    label = np.argwhere(label == 1)
    label = int(label)
    acc = 1 if np.argmax(out) == label else 0

    # out: vertor of probability
    # loss: num
    # acc: 1 or 0
    return out, loss, acc


def train(im, label, lr=.005):
    # Forward
    out, loss, acc = selfCNN(im, label)

    # gradient: loss 对于 softmax 输出层的 gradient
    gradient = np.zeros(10)
    label = np.argwhere(label == 1)
    label = int(label)
    # 只修改 label 值对应的
    gradient[label] = -1 / out[label]

    # Backprop
    # gradient：loss 对于 softmax 输入层的 gradient
    # 输入为 loss 对于 softmax 输出层的 gradient
    gradient = softmax.backprop(gradient, lr)
    # gradient：loss 对于池化层输入层的 gradient
    # 输入为 loss 对于池化层输出层的 gradient
    gradient = pool.backprop(gradient)
    # gradient：loss 对于卷基层输入层的 gradient
    # 输入为 loss 对于卷基层输出层的 gradient
    gradient = conv.backprop(gradient, lr)

    return loss, acc


from struct import unpack
import gzip


def __read_image(path):
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)
    return img


def __read_label(path):
    with gzip.open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.frombuffer(f.read(), dtype=np.uint8)
        # print(lab[1])
    return lab


# def __normalize_image(image):
#     img = image.astype(np.float32) / 255.0
#     return img


def __one_hot_label(label):
    lab = np.zeros((label.size, 10))
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab


def load_mnist(x_train_path, y_train_path, x_test_path, y_test_path, normalize=True, one_hot=True):
    '''读入MNIST数据集
    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label :
        one_hot为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    Returns
    ----------
    (训练图像, 训练标签), (测试图像, 测试标签)
    '''
    image = {
        'train': __read_image(x_train_path),
        'test': __read_image(x_test_path)
    }

    label = {
        'train': __read_label(y_train_path),
        'test': __read_label(y_test_path)
    }

    # if normalize:
    #     for key in ('train', 'test'):
    #         image[key] = __normalize_image(image[key])

    if one_hot:
        for key in ('train', 'test'):
            label[key] = __one_hot_label(label[key])

    return (image['train'], label['train']), (image['test'], label['test'])


x_train_path = './mnist/train-images-idx3-ubyte.gz'
y_train_path = './mnist/train-labels-idx1-ubyte.gz'
x_test_path = './mnist/t10k-images-idx3-ubyte.gz'
y_test_path = './mnist/t10k-labels-idx1-ubyte.gz'
(x_train, y_train), (x_test, y_test) = load_mnist(x_train_path, y_train_path, x_test_path, y_test_path)
x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)

# We only use the first 1k examples of each set in the interest of time.
# Feel free to change this if you want.
train_images = x_train[:1000]
train_labels = y_train[:1000]
test_images = x_test[:1000]
test_labels = y_test[:1000]

conv = Conv3x3(8)  # 28x28x1 -> 26x26x8
pool = MaxPool2()  # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10)  # 13x13x8 -> 10


print('MNIST CNN initialized!')

# Train the CNN for 3 epochs
for epoch in range(3):
    print('--- Epoch %d ---' % (epoch + 1))

    # Shuffle the training data
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    # Train
    loss = 0
    num_correct = 0
    # i: index
    # im: image
    # label: label
    # enumerate 函数用来增加索引值

    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i > 0 and i % 100 == 99:
            print(
                '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                (i + 1, loss / 100, num_correct)
            )
            loss = 0
            num_correct = 0

        l, acc = train(im, label)
        loss += 1
        num_correct += acc

# Test the CNN
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
    _, l, acc = selfCNN(im, label)
    loss += l
    num_correct += acc

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)