#! /usr/bin/env python
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import mnist

from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.pool import pool_2d
srng = RandomStreams(use_cuda = True)

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def convlayer(X, w, first=False):
    if first:
        conv = conv2d(X,w, border_mode='half') #half perfoms as scipy.signals 'full' for odd filter sizes
    else:
        conv = conv2d(X, w)
    conv = rectify(conv)
    max_pool = pool_2d(conv, ds=(2,2), ignore_border=False,mode='max')
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

def model(X,w_h_1, w_h_2,w_h_3, w_h2, w_o, p_use_input, p_use_hidden):
    X = dropout(X, p_use_input)
    h = convlayer(X, w_h_1)
    h = dropout(h, p_use_hidden)
    h_1 = convlayer(h, w_h_2)
    h_1 = dropout(h_1, p_use_hidden)
    h_2 = convlayer(h_1, w_h_3)
    h_2 = T.flatten(h_2,outdim=2)
    h2 = rectify(T.dot(h_2, w_h2))
    h2 = dropout(h2, p_use_hidden)
    py_x = softmax(T.dot(h2, w_o))
    return h, h_1, h_2, h2, py_x

trX, teX, trY, teY = mnist(onehot=True)
trX = trX.reshape(-1,1,28,28) #training data
teX = teX.reshape(-1,1,28,28)
X = T.ftensor4()
Y = T.ftensor4()


w_h_1 = init_weights((32,1,5,5))
w_h_2 = init_weights((64,32,5,5))
w_h_3 = init_weights((128,64,2,2))
w_h2 = init_weights((128, 625))
w_o = init_weights((625, 10))

params = [w_h_1, w_h_2,w_h_3, w_h2, w_o]

noise_h,noise_h_1, noise_h_2, noise_h2, noise_py_x = model(X, w_h_1, w_h_2,w_h_3, w_h2, w_o, 0.8, 0.5)
h, h_1, h_2, h2, py_x = model(X, w_h_1, w_h_2,w_h_3, w_h2, w_o, 1., 1.)
y_x = T.argmax(py_x, axis=1)
cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w_h, w_h2, w_o]
updates = RMSprop(cost, params, lr=0.001)
train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
for i in range(1): #you can adjust this if training takes too long
    for start, end in zip(list(range(0, len(trX), 128)), list(range(128, len(trX), 128))):
        cost = train(trX[start:end], trY[start:end])
    print(np.mean(np.argmax(teY, axis=1) == predict(teX)))
