#! /usr/bin/env python
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import mnist

from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.pool import pool_2d

theano.config.assert_no_cpu='warn'
theano.config.exception_verbosity='high'

BSIZE = 512

srng = RandomStreams(use_cuda = True)
print('start')

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape, name=None):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01),
                         name=name)

def rectify(X):
    return T.maximum(X, 0.)

def convlayer(X, w, first=False):
    if first:
        #half perfoms as scipy.signals 'full' for odd filter sizes
        conv = conv2d(X,w, border_mode='full')
    else:
        conv = conv2d(X, w)
    nonl = rectify(conv)
    max_pool = pool_2d(nonl, ds=(2,2), ignore_border=False, mode='max')
    return max_pool

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates


def dropout(X, p=1):
    mask = srng.binomial(size=X.shape, p=p, dtype='floatX')
    return X * mask / p

def model(X, w_c_1, w_c_2, w_c_3, w_f, w_o, p_use_input, p_use_hidden):
    X = dropout(X, p_use_input)
    c_1 = convlayer(X, w_c_1, first=False)
    c_1 = dropout(c_1, p_use_hidden)
    c_2 = convlayer(c_1, w_c_2)
    c_2 = dropout(c_2, p_use_hidden)
    c_3 = convlayer(c_2, w_c_3)
    c_3 = T.flatten(c_3, outdim=2)
    f = rectify(T.dot(c_3, w_f))
    f = dropout(f, p_use_hidden)
    py_x = softmax(T.dot(f, w_o))
    return c_1, c_2, c_3, f, py_x

trX, teX, trY, teY = mnist(onehot=True)
trX = trX.reshape(-1,1,28,28) #training data
teX = teX.reshape(-1,1,28,28)
X = T.ftensor4()
Y = T.fmatrix()

w_c_1 = init_weights((32,1,5,5), 'w_c_1')
w_c_2 = init_weights((64,32,5,5), 'w_c_2')
w_c_3 = init_weights((128,64,2,2), 'w_c_3')
w_f = init_weights((512, 625), 'w_f')
w_o = init_weights((625, 10), 'w_o')


noise_h,noise_h_1, noise_h_2, noise_h2, noise_py_x = model(
        X, w_c_1, w_c_2,w_c_3, w_f, w_o, 0.8, 0.5)

h, h_1, h_2, h2, py_x = model(
        X, w_c_1, w_c_2,w_c_3, w_f, w_o, 1., 1.)
y_x = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w_c_1, w_c_2, w_c_3, w_f, w_o]
updates = RMSprop(cost, params, lr=0.001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

for i in range(20): #you can adjust this if training takes too long
    for start, end in zip(list(range(0, len(trX), BSIZE)), list(range(BSIZE, len(trX), BSIZE))):
        cost = train(trX[start:end], trY[start:end])
    print(np.mean(np.argmax(teY, axis=1) == predict(teX)))
