import numpy as np
import math
import scipy.io
import copy
import matplotlib.pyplot as plt

def load_mnist(filename, fullset=False, debug = True, shuffle = False, transpose = True):
    # load the data from the txt file, if fullset sets to true,
    # load the entire dataset, else, only load first 100 examples
    # to do sanity check
    xtrain = np.loadtxt("data/"+filename+"train.txt", delimiter=',', usecols=range(0,784))
    ytrain = np.loadtxt("data/"+filename+"train.txt", delimiter=',', usecols=(784,), dtype=np.int64)

    xtest = np.loadtxt("data/" + filename + "test.txt", delimiter=',', usecols=range(0, 784))
    ytest = np.loadtxt("data/" + filename + "test.txt", delimiter=',', usecols=(784,), dtype=np.int64)

    xvalidate = np.loadtxt("data/" + filename + "valid.txt", delimiter=',', usecols=range(0, 784))
    yvalidate = np.loadtxt("data/" + filename + "valid.txt", delimiter=',', usecols=(784,), dtype=np.int64)

    if not fullset:
        xtrain = xtrain[0:500,:]
        ytrain = ytrain[0:500]

    if debug:
        print "shape of xtrain: " + str(xtrain.shape)
        print "shape of ytrain: " + str(ytrain.shape)
        print "shape of xtest: " + str(xtest.shape)
        print "shape of ytest: " + str(ytest.shape)
        print "shape of xvalid: " + str(xvalidate.shape)
        print "shape of yvalid: " + str(yvalidate.shape)

    # the dataset is listed in order of the digit value, so to perform
    # SGD, we need to shuffle the data to emulate a random data generation
    if shuffle:
        np.random.seed(10)
        train_indices = range(xtrain.shape[0])
        np.random.shuffle(train_indices)
        xtrain.take(train_indices, axis=0, out=xtrain)
        ytrain.take(train_indices, axis=0, out=ytrain)
        test_indices = range(xtest.shape[0])
        np.random.shuffle(test_indices)
        xtest.take(test_indices, axis=0, out=xtest)
        ytest.take(test_indices, axis=0, out=ytest)

    if transpose:
        xtrain = xtrain.T
        xtest = xtest.T
        xvalidate = xvalidate.T

    return [xtrain, ytrain, xvalidate, yvalidate, xtest, ytest]


# this function takes argument layers, which is a dictionary that
# describe the layer's properties and initialize hyperparameters
# for each layer
def init_neuralnet(layers):
    params = {}
    h = layers[1]['height']
    w = layers[1]['width']

    for i in range(2, len(layers)+1):
        params[i - 1] = {}
        if layers[i]['type'] == 'IP':
            # the number of weight parameter is 784*count(perceptions)
            scale = np.sqrt(6. / ((h * w) + layers[i]['num']))
            sample_size = h*w
            params[i-1]['w'] = 2*scale*np.random.rand(sample_size, layers[i]['num'])-scale
            params[i-1]['b'] = np.zeros(layers[i]['num'])
            h = layers[i]['num']
            w = 1
        if layers[i]['type'] == 'SOFTMAX':
            params[i-1]['w'] = np.array([])
            params[i-1]['b'] = np.array([])

        if layers[i]['type'] == 'SIGMOID':
            params[i-1]['w'] = np.array([])
            params[i-1]['b'] = np.array([])

        if layers[i]['type'] == 'LOSS':
            # final layer, only takes the input from the softmax
            params[i - 1]['w'] = np.array([])
            params[i - 1]['b'] = np.array([])

    return params


def train_model(params, layers, data, labels):
    """

    Args:
      params: a dictionary that stores hyper parameters
      layers: a dictionary that defines LeNet
      data: input data with shape (784, batch size)
      labels: label with shape (batch size,)

    Returns:
      cp: train accuracy for the train data
      param_grad: gradients of all the layers whose parameters are stored in params

    """
    l = len(layers)
    batch_size = layers[1]['batch_size']
    assert layers[1]['type'] == 'DATA', 'first layer must be data layer'

    # output is the dictionary that stores the data in back propagation
    param_grad = {}
    cp = {}
    output = {}
    output[1] = {}
    output[1]['data'] = data
    output[1]['height'] = layers[1]['height']
    output[1]['width'] = layers[1]['width']
    output[1]['batch_size'] = layers[1]['batch_size']
    output[1]['diff'] = 0

    for i in range(2, l):
        if layers[i]['type'] == 'IP':
            output[i] = inner_product_forward(output[i - 1], layers[i], params[i - 1])
        elif layers[i]['type'] == 'SOFTMAX':
            output[i] = softmax_forward(output[i - 1], layers[i])
        elif layers[i]['type'] == 'SIGMOID':
            output[i] = sigmoid_forward(output[i - 1], layers[i])

    i = l
    assert layers[i]['type'] == 'LOSS', 'last layer must be loss layer'

    #calculate the loss function
    [input_od, percent, label_predict] = lossfunction(output[i - 1]['data'],labels, layers[i]['num'], 1)
    for i in range(l, 1, -1):
        param_grad[i - 1] = {}
        if layers[i]['type'] == 'IP':
            output[i]['diff'] = input_od
            param_grad[i - 1], input_od = inner_product_backward(output[i],
                                                             output[i - 1],
                                                             layers[i],
                                                             params[i - 1])
        elif layers[i]['type'] == 'SOFTMAX': # gradient w.r.t to the input od softmax is
                                           # calculated in lossfunction

            param_grad[i - 1]['w'] = np.array([])
            param_grad[i - 1]['b'] = np.array([])

        elif layers[i]['type'] == 'SIGMOID':
            output[i]['diff'] = input_od
            input_od = sigmoid_backward(output[i], output[i - 1], layers[i])
            param_grad[i - 1]['w'] = np.array([])
            param_grad[i - 1]['b'] = np.array([])

        elif layers[i]['type'] == 'LOSS':
            param_grad[i - 1]['w'] = np.array([])
            param_grad[i - 1]['b'] = np.array([])

        param_grad[i - 1]['w'] = param_grad[i - 1]['w'] / batch_size
        param_grad[i - 1]['b'] = param_grad[i - 1]['b'] / batch_size

    #back propogation

    return [percent, param_grad]

def inner_product_forward(input, layer, param):
    """Fully connected layer forward

    Args:
    input: a dictionary contains input data and shape information
    layer: one cnn layer, defined in testLeNet.py
    param: parameters, a dictionary

    Returns:
    output: a dictionary contains output data and shape information
    """
    num = layer['num']
    batch_size = input['batch_size']

    output = {}
    output['height'] = 1
    output['width'] = 1
    output['channel'] = num
    output['batch_size'] = batch_size
    output['data'] = np.zeros((num, batch_size))


    for n in range(batch_size):
      input_n = input['data'][:, n]
      tmp_output = input_n.dot(param['w']) + param['b']
      output['data'][:, n] = tmp_output.flatten()
    assert np.all(output['data'].shape == (num, batch_size)), 'output[\'data\'] has incorrect shape!'
    return output


def inner_product_backward(output, input, layer, param):
    """Fully connected layer backward

  Args:
    output: a dictionary contains output data and shape information
    input: a dictionary contains input data and shape information
    layer: one cnn layer, defined in testLeNet.py
    param: parameters, a dictionary

  Returns:
    para_grad: a dictionary stores gradients of parameters
    input_od: gradients w.r.t input data
  """
    param_grad = {}
    param_grad['b'] = np.zeros(param['b'].shape)
    param_grad['w'] = np.zeros(param['w'].shape)
    input_od = np.zeros(input['data'].shape)
    w = param['w']
    input_od = np.dot(w, output['diff'])
    output_od = output['diff'].transpose()
    param_grad['w'] = np.dot(input['data'], output_od)
    param_grad['b'] = np.sum(output['diff'], axis=1)

    # implementation ends

    assert np.all(input['data'].shape == input_od.shape), 'input_od has incorrect shape!'

    return param_grad, input_od


def softmax_forward(input, layer):
    """softmax foward

      Args:
        input: a dictionary contains input data and shape information
        layer: one cnn layer, defined in testLeNet.py

      Returns:
        output: a dictionary contains output data and shape information
      """
    output = {}
    output['height'] = input['height']
    output['width'] = input['width']
    output['channel'] = input['channel']
    output['batch_size'] = input['batch_size']
    output['data'] = np.zeros(input['data'].shape)
    output['data'] = softmax(input['data'])
    # implementation en
    assert np.all(output['data'].shape == input['data'].shape), 'output[\'data\'] has incorrect shape!'
    return output
    return -1

#   this function is referenced from http://cs231n.github.io/linear-classify/#softmax
#   to deal with Numeric instability
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=0))
    return e_x / e_x.sum(axis=0) #sum column-wise

def sigmoid_forward(input, layer):
    """sigmoid foward

      Args:
        input: a dictionary contains input data and shape information
        layer: one cnn layer, defined in testLeNet.py

      Returns:
        output: a dictionary contains output data and shape information
      """
    output = {}
    output['height'] = input['height']
    output['width'] = input['width']
    output['channel'] = input['channel']
    output['batch_size'] = input['batch_size']
    output['data'] = np.zeros(input['data'].shape)

    # TODO: implement your relu forward pass here
    # implementation begins
    # test: input['data'][0][0] = 1
    #zero_matrix = np.zeros(input['data'].shape)
    output['data'] = sigmoid(input['data'])
    # implementation ends

    assert np.all(output['data'].shape == input['data'].shape), 'output[\'data\'] has incorrect shape!'

    return output

def sigmoid_backward(output, input, layer):
    input_od = np.zeros(input['data'].shape)
    input_od = dsigmoid(output['data'])* output['diff']
    return input_od

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# derivative of sigmoid
# sigmoid(x) * (1.0 - sigmoid(x))
# the way we use this y is already sigmoided

def dsigmoid(y):
    return y * (1.0 - y)

def lossfunction(prob, y, K, prediction):
    batch_size = prob.shape[1]
    I = np.zeros((K, batch_size))
    I[y, np.arange(batch_size)] = 1
    nll = 0
    od = np.zeros(prob.shape)


    if prediction == 1:
        indices = np.argmax(prob, axis=0)
        percent = len(np.where(y == indices)[0]) / float(len(y))
    else:
        percent = 0

    # compute gradients
    od = prob - I

    return  od, percent, indices

def get_lr(step, epsilon, gamma, power):
    """Get the learning rate at step iter"""
    lr_t = epsilon / math.pow(1 + gamma * step, power)
    return lr_t

def sgd_momentum(w_rate, b_rate, mu, decay, params, param_winc, param_grad):
    """Update the parameters with sgd with momentum

  Args:
    w_rate (scalar): sgd rate for updating w
    b_rate (scalar): sgd rate for updating b
    mu (scalar): momentum
    decay (scalar): weight decay of w
    params (dictionary): original weight parameters
    param_winc (dictionary): buffer to store history gradient accumulation
    param_grad (dictionary): gradient of parameter

  Returns:
    params_ (dictionary): updated parameters
    param_winc_ (dictionary): gradient buffer of previous step
  """

    params_ = copy.deepcopy(params)
    param_winc_ = copy.deepcopy(param_winc)

    # TODO: your implementation goes below this comment
    # implementation begins
    for i in range(len(params)-1):
        i = i + 1
        param_winc_[i]['w'] = mu * param_winc_[i]['w'] + w_rate * (param_grad[i]['w'] + decay * params_[i]['w'])
        param_winc_[i]['b'] = mu * param_winc_[i]['b'] + b_rate * param_grad[i]['b']
        params_[i]['w'] = params_[i]['w'] - param_winc_[i]['w']
        params_[i]['b'] = params_[i]['b'] - param_winc_[i]['b']
    # implementation ends

    assert len(params_) == len(param_grad), 'params_ does not have the right length'
    assert len(param_winc_) == len(param_grad), 'param_winc_ does not have the right length'

    return params_, param_winc_