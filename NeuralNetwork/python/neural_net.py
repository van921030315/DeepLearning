import numpy as np
import copy


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


def init_neuralnet(layers):
    params = {}

    h = layers[1]['height']
    w = layers[1]['width']
    c = layers[1]['channel']

    # Starts with second layer, first layer is data layer
    for i in range(2, len(layers) + 1):
        params[i - 1] = {}
        if layers[i]['type'] == 'IP':
            scale = np.sqrt(
                6. / (h*w*c + layers[i]['num']))
            params[i - 1]['w'] = 2 * scale * np.random.rand(h * w * c, layers[i]['num']) - scale
            params[i - 1]['b'] = np.zeros(layers[i]['num'])
            h = 1
            w = 1
            c = layers[i]['num']
        elif layers[i]['type'] == 'RELU':
            params[i - 1]['w'] = np.array([])
            params[i - 1]['b'] = np.array([])
        elif layers[i]['type'] == 'SIGMOID':
            params[i - 1]['w'] = np.array([])
            params[i - 1]['b'] = np.array([])
        elif layers[i]['type'] == 'LOSS':
            scale = np.sqrt(
                6. / ((h*w*c + layers[i]['num'])))
            num = layers[i]['num']
            # last layer is K-1
            params[i - 1]['w'] = 2 * scale * np.random.rand(h * w * c, num - 1) - scale
            params[i - 1]['b'] = np.zeros(num - 1)
            h = 1
            w = 1
            c = layers[i]['num']
    return params

def neural_net(params, layers, data, labels):

    l = len(layers)
    batch_size = layers[1]['batch_size']
    assert layers[1]['type'] == 'DATA', 'first layer must be data layer'

    param_grad = {}
    cp = {}
    output = {}
    output[1] = {}
    output[1]['data'] = data
    output[1]['height'] = layers[1]['height']
    output[1]['width'] = layers[1]['width']
    output[1]['channel'] = layers[1]['channel']
    output[1]['batch_size'] = layers[1]['batch_size']
    output[1]['diff'] = 0

    for i in range(2, l):
        if layers[i]['type'] == 'IP':
            output[i] = inner_product_forward(output[i - 1], layers[i], params[i - 1])
        elif layers[i]['type'] == 'RELU':
            output[i] = relu_forward(output[i - 1], layers[i])
        elif layers[i]['type'] == 'SIGMOID':
            output[i] = sigmoid_forward(output[i-1], layers[i])

    i = l
    assert layers[i]['type'] == 'LOSS', 'last layer must be loss layer'

    wb = np.vstack([params[i - 1]['w'], params[i - 1]['b']])
    [cost, grad, input_od, percent] = mlrloss(wb,
                                              output[i - 1]['data'],
                                              labels,
                                              layers[i]['num'], 1)
    param_grad[i - 1] = {}
    param_grad[i - 1]['w'] = grad[0:-1, :]
    param_grad[i - 1]['b'] = grad[-1, :]
    param_grad[i - 1]['w'] = param_grad[i - 1]['w'] / batch_size
    param_grad[i - 1]['b'] = param_grad[i - 1]['b'] / batch_size

    cp['cost'] = cost / batch_size
    cp['percent'] = percent

    # range: [l-1, 2]
    for i in range(l - 1, 1, -1):
        param_grad[i - 1] = {}
        if layers[i]['type'] == 'IP':
            output[i]['diff'] = input_od
            param_grad[i - 1], input_od = inner_product_backward(output[i],
                                                                 output[i - 1],
                                                                 layers[i],
                                                                 params[i - 1])
        elif layers[i]['type'] == 'RELU':
            output[i]['diff'] = input_od
            input_od = relu_backward(output[i], output[i - 1], layers[i])
            param_grad[i - 1]['w'] = np.array([])
            param_grad[i - 1]['b'] = np.array([])

        elif layers[i]['type'] == 'SIGMOID':
            output[i]['diff'] = input_od
            input_od = sigmoid_backward(output[i], output[i - 1], layers[i])
            param_grad[i - 1]['w'] = np.array([])
            param_grad[i - 1]['b'] = np.array([])

        param_grad[i - 1]['w'] = param_grad[i - 1]['w'] / batch_size
        param_grad[i - 1]['b'] = param_grad[i - 1]['b'] / batch_size

    return cp, param_grad

def predict(params, layers, data, labels):
    l = len(layers)
    batch_size = layers[1]['batch_size']
    assert layers[1]['type'] == 'DATA', 'first layer must be data layer'
    cp = {}
    output = {}
    output[1] = {}
    output[1]['data'] = data
    output[1]['height'] = layers[1]['height']
    output[1]['width'] = layers[1]['width']
    output[1]['channel'] = layers[1]['channel']
    output[1]['batch_size'] = layers[1]['batch_size']
    output[1]['diff'] = 0

    for i in range(2, l):
        if layers[i]['type'] == 'IP':
            output[i] = inner_product_forward(output[i - 1], layers[i], params[i - 1])
        elif layers[i]['type'] == 'RELU':
            output[i] = relu_forward(output[i - 1], layers[i])
        elif layers[i]['type'] == 'SIGMOID':
            output[i] = sigmoid_forward(output[i - 1], layers[i])

    i = l
    assert layers[i]['type'] == 'LOSS', 'last layer must be loss layer'

    wb = np.vstack([params[i - 1]['w'], params[i - 1]['b']])
    [cost, _, _, percent] = mlrloss(wb,
                                              output[i - 1]['data'],
                                              labels,
                                              layers[i]['num'], 1)

    cp['cost'] = cost / batch_size
    cp['percent'] = percent
    return cp, output


def relu_forward(input, layer):
    output = {}
    output['height'] = input['height']
    output['width'] = input['width']
    output['channel'] = input['channel']
    output['batch_size'] = input['batch_size']
    output['data'] = np.zeros(input['data'].shape)

    zero_matrix = np.zeros(input['data'].shape)
    output['data'] = np.fmax(input['data'], zero_matrix)
    assert np.all(output['data'].shape == input['data'].shape), 'output[\'data\'] has incorrect shape!'

    return output

def relu_backward(output, input, layer):
    input_od = np.where(input['data'] >= 0, output['diff'], 0)
    assert np.all(input['data'].shape == input_od.shape), 'input_od has incorrect shape!'

    return input_od

def sigmoid_forward(input, layer):
    output = {}
    output['height'] = input['height']
    output['width'] = input['width']
    output['channel'] = input['channel']
    output['batch_size'] = input['batch_size']

    output['data'] = sigmoid(input['data'])
    assert np.all(output['data'].shape == input['data'].shape), 'output[\'data\'] has incorrect shape!'

    return output

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_backward(output, input, layer):
    input_od = dsigmoid(output['data'])*output['diff']
    #input_od_approx = finite_difference(output, input, 1e-4 )
    assert np.all(input['data'].shape == input_od.shape), 'input_od has incorrect shape!'
    return input_od

def dsigmoid(y):
    return y * (1.0 - y)

def inner_product_forward(input, layer, param):
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
    param_grad = {}
    param_grad['b'] = np.zeros(param['b'].shape)
    param_grad['w'] = np.zeros(param['w'].shape)
    input_od = np.zeros(input['data'].shape)
    w = param['w']
    input_od = np.dot(w, output['diff'])
    output_od = output['diff'].transpose()
    param_grad['w'] = np.dot(input['data'], output_od)
    param_grad['b'] = np.sum(output['diff'], axis=1)

    assert np.all(input['data'].shape == input_od.shape), 'input_od has incorrect shape!'

    return param_grad, input_od


def finite_difference(output, input, h):
    x_plus_h = input
    x_plus_h['data'] = x_plus_h['data'] + h
    layer = 0
    fx_plus_h = sigmoid_forward(x_plus_h, layer)
    input_od_approx = np.multiply(((fx_plus_h['data'] - output['data']) / h), output['diff'])
    return input_od_approx

def sigmoid_forward(input, layer):
    output = {}
    output['height'] = input['height']
    output['width'] = input['width']
    output['channel'] = input['channel']
    output['batch_size'] = input['batch_size']
    output['data'] = np.zeros(input['data'].shape)

    output['data'] = sigmoid(input['data'])
        # implementation ends

    assert np.all(output['data'].shape == input['data'].shape), 'output[\'data\'] has incorrect shape!'
    return output



def softmax_forward(input):
    """softmax foward

      Args:
        input: a dictionary contains input data and shape information
        layer: one cnn layer, defined in testLeNet.py

      Returns:
        output: a dictionary contains output data and shape information
      """
    output = {}
    output['height'] = 1
    output['width'] = 1
    output['channel'] = input.shape[0]
    output['batch_size'] = input.shape[1]
    output['data'] = np.zeros(input.shape)
    output['data'] = softmax(input)
    # implementation en
    assert np.all(output['data'].shape == input.shape), 'output[\'data\'] has incorrect shape!'
    return output['data']

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=0))
    return e_x / e_x.sum(axis=0) #sum column-wise

def mlrloss(wb, X, y, K, prediction):
    (_, batch_size) = X.shape
    theta = wb[:-1, :]
    bias = wb[-1, :]

    # Convert ground truth label to one-hot vectors
    I = np.zeros((K, batch_size))
    I[y, np.arange(batch_size)] = 1

    # Compute the values after the linear transform
    activation = np.transpose(X.T.dot(theta) + bias)
    activation = np.vstack([activation, np.zeros(batch_size)])

    prob = softmax_forward(activation)

    nll = 0
    od = np.zeros(prob.shape)
    nll = -np.sum(np.log(prob[y, np.arange(batch_size)]))

    if prediction == 1:
        indices = np.argmax(prob, axis=0)
        percent = len(np.where(y == indices)[0]) / float(len(y))
    else:
        percent = 0

    # compute gradients
    od = prob - I
    gw = od.dot(X.T)
    gw = gw[0:-1, :].T
    gb = np.sum(od, axis=1)
    gb = gb[0:-1]
    g = np.vstack([gw, gb])
    od = theta.dot(od[0:-1, :])
    return nll, g, od, percent


def sgd_momentum(w_rate, b_rate, mu, decay, params, param_winc, param_grad):
    params_ = copy.deepcopy(params)
    param_winc_ = copy.deepcopy(param_winc)

    for i in range(len(params)):
        i = i + 1
        param_winc_[i]['w'] = mu * param_winc_[i]['w'] + w_rate * (param_grad[i]['w'] + decay * params_[i]['w'])
        param_winc_[i]['b'] = mu * param_winc_[i]['b'] + b_rate * param_grad[i]['b']
        params_[i]['w'] = params_[i]['w'] - param_winc_[i]['w']
        params_[i]['b'] = params_[i]['b'] - param_winc_[i]['b']

    return params_, param_winc_