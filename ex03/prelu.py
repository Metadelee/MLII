#! /usr/bin/env python
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import mnist

srng = RandomStreams(use_cuda = True)

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def init_slope(shape):
    return theano.shared(floatX(0.25 + np.zeros((1,shape))),                        broadcastable=(True, False))

def rectify(X):
    return T.maximum(X, 0.)

def PRelu(X, a):
    return T.maximum(X, 0) + a * T.minimum(X, 0)

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


def model(X, w_h, w_h2, w_o, s_h, s_h2, p_use_input, p_use_hidden):
    #X = dropout(X, p_use_input)
    #h = rectify(T.dot(X, w_h))
    h = PRelu(T.dot(X, w_h), s_h)
    #h = dropout(h, p_use_hidden)
    #h2 = rectify(T.dot(h, w_h2))
    h2 = PRelu(T.dot(X, w_h2), s_h2)
    h2 = dropout(h2, p_use_hidden)
    py_x = softmax(T.dot(h2, w_o))
    return h, h2, py_x

trX, teX, trY, teY = mnist(onehot=True)

X = T.fmatrix()
Y = T.fmatrix()

w_h = init_weights((784, 625))
w_h2 = init_weights((625, 625))
w_o = init_weights((625, 10))

s_h = init_slope(625)
s_h2 = init_slope(625)

noise_h, noise_h2, noise_py_x = model(X, w_h, w_h2, w_o, s_h, s_h2, 0.8, 0.5)
h, h2, py_x = model(X, w_h, w_h2, w_o, s_h, s_h2, 1., 1.)
y_x = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w_h, w_h2, w_o,]
updates = RMSprop(cost, params, lr=0.001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

for i in range(100): #you can adjust this if training takes too long
    for start, end in zip(list(range(0, len(trX), 128)), list(range(128, len(trX), 128))):
        cost = train(trX[start:end], trY[start:end])
    print(np.mean(np.argmax(teY, axis=1) == predict(teX)))

