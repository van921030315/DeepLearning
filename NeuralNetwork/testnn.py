import numpy as np
import nn_mnist
import pickle
import copy
import random
import matplotlib.pyplot as plt

def getneuralnet():
    layers = {}
    # The first layer is a data layer,
    layers[1] = {}
    layers[1]['type'] = 'DATA'
    layers[1]['height'] = 784
    layers[1]['width'] = 1
    layers[1]['channel'] = 1
    layers[1]['batch_size'] = 1

    # The linear production layer, nums is the
    # number of hidden units in this layer
    layers[2] = {}
    layers[2]['type'] = 'IP'
    layers[2]['num'] = 100

    # Activation function for hidden leayer
    layers[3] = {}
    layers[3] ['type'] = 'SIGMOID'

    layers[4] = {}
    layers[4]['type'] = 'IP'
    layers[4]['num'] = 10

    layers[5] = {}
    layers[5]['type'] = 'SOFTMAX'

    layers[6] = {}
    layers[6]['type'] = 'LOSS'
    layers[6]['num'] = 10

    return layers


def train_net(params, layers, data, labels):

    assert (layers[1]['type'] == 'DATA')
    l = len(layers)
    batch_size = 200
    if 'batch_size' in layers[1]:
        batch_size = layers[1]['batch_size']
    output = {}
    # data is the original pixel data feed to the input
    # data shape is (784, batch_size)
    output[1]['data'] = data
    output[1]['height'] = layers[1]['height']
    output[1]['width'] = layers[1]['width']

def plot(data, height, width):
    img = data.reshape(28, 28)
    plt.matshow(img)

def main():
    xtrain, ytrain, xval, yval, xtest, ytest = nn_mnist.load_mnist("digits", shuffle=True, fullset=True)
    # xtrain returns (784, samples) ytrain returns samples)
    #xtrain = np.hstack([xtrain, xval])
    #ytrain = np.hstack([ytrain, yval])
    max_iter = 30000;
    mu = 0.9
    epsilon = 0.01
    gamma = 0.0001
    power = 0.75
    weight_decay = 0.0005
    w_lr = 0.1
    b_lr = 0.1

    layers = getneuralnet()
    params = nn_mnist.init_neuralnet(layers)
    param_winc = copy.deepcopy(params)

    for l_idx in range(1, len(layers)):
        param_winc[l_idx]['w'] = np.zeros(param_winc[l_idx]['w'].shape)
        param_winc[l_idx]['b'] = np.zeros(param_winc[l_idx]['b'].shape)
    # try plot a picture

    m_train = xtrain.shape[1]
    indices = range(m_train)
    random.shuffle(indices)
    batch_size = layers[1]['batch_size']
    # learning eate
    for step in range(max_iter):
        start_idx = step * batch_size % m_train
        end_idx = (step + 1) * batch_size % m_train
        if start_idx > end_idx:
            random.shuffle(indices)
            continue

        idx = indices[start_idx: start_idx+batch_size]
        [percent, param_grad] = nn_mnist.train_model(params, layers, xtrain[:, idx] , ytrain[idx])

        w_rate = nn_mnist.get_lr(step, epsilon * w_lr, gamma, power)
        b_rate = nn_mnist.get_lr(step, epsilon * b_lr, gamma, power)


        params, param_winc = nn_mnist.sgd_momentum(w_rate,
                                                    b_rate,
                                                    mu,
                                                    weight_decay,
                                                    params,
                                                    param_winc,
                                                    param_grad)

        if step % 500 == 0:
            layers[1]['batch_size'] = xval.shape[1]
            percent, _ = nn_mnist.train_model(params, layers, xval, yval)
            print '\nvalidation accuracy: %f\n' % percent
            layers[1]['batch_size'] = 20

    layers[1]['batch_size'] = xtrain.shape[1]
    percent, _ = nn_mnist.train_model(params, layers, xtrain, ytrain)
    layers[1]['batch_size'] = xtest.shape[1]
    percent, _ = nn_mnist.train_model(params, layers, xtest, ytest)
    print '\ntest accuracy: %f\n' % percent




if __name__ == '__main__':
  main()
