import numpy as np
import neural_net
import pickle
import copy
import random
import matplotlib.pyplot as plt
import sys

def get_net(layer = 1, activation = 'SIGMOID'):

  if layer == 1:
      layers = {}
      layers[1] = {}
      layers[1]['type'] = 'DATA'
      layers[1]['height'] = 784
      layers[1]['width'] = 1
      layers[1]['channel'] = 1
      layers[1]['batch_size'] = 100

      layers[2] = {}
      layers[2]['type'] = 'IP'
      layers[2]['num'] = 100
      layers[2]['init_type'] = 'uniform'

      layers[3] = {}
      layers[3]['type'] = activation

      layers[4] = {}
      layers[4]['type'] = 'LOSS'
      layers[4]['num'] = 10
  elif layer == 2:
      layers = {}
      layers[1] = {}
      layers[1]['type'] = 'DATA'
      layers[1]['height'] = 784
      layers[1]['width'] = 1
      layers[1]['channel'] = 1
      layers[1]['batch_size'] = 100

      layers[2] = {}
      layers[2]['type'] = 'IP'
      layers[2]['num'] = 100
      layers[2]['init_type'] = 'uniform'

      layers[3] = {}
      layers[3]['type'] = activation

      layers[4] = {}
      layers[4]['type'] = 'IP'
      layers[4]['num'] = 100
      layers[4]['init_type'] = 'uniform'

      layers[5] = {}
      layers[5]['type'] = activation

      layers[6] = {}
      layers[6]['type'] = 'LOSS'
      layers[6]['num'] = 10


  return layers


def main(argv):
  l = 1
  a = 'SIGMOID'
  for i in range(len(argv)):
      print argv[i]
      if argv[i] == "layer1":
          l = 1
      elif argv[i] == "layer2":
          l = 2
      elif argv[i] == "sigmoid":
          a = 'SIGMOID'
      elif sys.argv[i] == "relu":
          a = 'RELU'
  # define lenet
  #layers = get_net()
  # to test 2-layer network or change activation function
  # uncomment the below comments
  layers = get_net(layer=l, activation=a)
  # layers = get_net(activation='RELU')

  # load data
  # change the following value to true to load the entire dataset
  fullset = False
  xtrain, ytrain, xval, yval, xtest, ytest = neural_net.load_mnist("digits", fullset=True,  shuffle=True)

  m_train = xtrain.shape[1]

  # hyperparameters
  batch_size = 64
  mu = 0.91
  weight_decay = 0.0005
  w_lr = 0.3
  b_lr = 0.3

  # display setting
  test_interval = 200
  display_interval = 200
  max_iter = 50000

  # initialize parameters
  params = neural_net.init_neuralnet(layers)
  param_winc = copy.deepcopy(params)
  batch_size = layers[1]['batch_size']
  for l_idx in range(1, len(layers)):
    param_winc[l_idx]['w'] = np.zeros(param_winc[l_idx]['w'].shape)
    param_winc[l_idx]['b'] = np.zeros(param_winc[l_idx]['b'].shape)

  # learning iterations
  indices = range(m_train)
  random.shuffle(indices)
  for step in range(max_iter):
    # get mini-batch and setup the cnn with the mini-batch
    start_idx = step * batch_size % m_train
    end_idx = (step+1) * batch_size % m_train
    if start_idx > end_idx:
      random.shuffle(indices)
      continue
    idx = indices[start_idx: end_idx]

    [cp, param_grad] = neural_net.neural_net(params,
                                          layers,
                                          xtrain[:, idx],
                                          ytrain[idx])

    params, param_winc = neural_net.sgd_momentum(w_lr,
                           b_lr,
                           mu,
                           weight_decay,
                           params,
                           param_winc,
                           param_grad)

    # display training loss
    if (step+1) % display_interval == 0:
      print 'trainning: cross-entropy = %f percent = %f' % (cp['cost'], cp['percent'])

    # display vlidationn accuracy
    if (step+1) % test_interval == 0:
      batch_size_ = layers[1]['batch_size']
      layers[1]['batch_size'] = xval.shape[1]
      cpval, _ = neural_net.predict(params, layers, xval, yval)
      layers[1]['batch_size'] = batch_size_
      print 'validation: cross-entropy = %f percent = %f\n' % (cpval['cost'], cpval['percent'])


  layers[1]['batch_size'] = xtest.shape[1]
  cptest, _ = neural_net.predict(params, layers, xtest, ytest)
  print '\ntest accuracy: %f, cross-entropy = %f \n' % (cptest['percent'], cptest['cost'])
  layers[1]['batch_size'] = xtrain.shape[1]
  cptest, _ = neural_net.predict(params, layers, xtrain, ytrain)
  print '\ntrain accuracy: %f, cross-entropy = %f \n' % (cptest['percent'], cptest['cost'])
  layers[1]['batch_size'] = xval.shape[1]
  cptest, _ = neural_net.predict(params, layers, xval, yval)
  print '\nvalidation accuracy: %f, cross-entropy = %f \n' % (cptest['percent'], cptest['cost'])

  pickle_path = 'nn_params.mat'
  pickle_file = open(pickle_path, 'wb')
  pickle.dump(params, pickle_file)
  pickle_file.close()

  pickle_path = 'nn_layers.mat'
  pickle_file = open(pickle_path, 'wb')
  pickle.dump(layers, pickle_file)
  pickle_file.close()

if __name__ == '__main__':
  main(sys.argv[1:])
