# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
print(__file__)
path = os.path.dirname(os.path.abspath( __file__ ))
print(path)

import sys
print(sys.argv[0])

import numpy as np
np.log2

x = np.array([5,3,6,7,3])
w = np.array([.3,.6,.2,.5,.1])
z = np.dot(x,w)
print(z)

import time
a = np.random.rand(1000000)
b = np.random.rand(1000000)
# vectorised
t_st = time.time()
c = np.dot(a,b)
t_end = time.time()
print("Vector Calc: {:.2f} in {:.2f} msecs".format(c,(t_end-t_st)*1000))
# not vectorised
t_st2 = time.time()
c = 0
for i in range(1000000):
    c += a[i]*b[i]
t_end2 = time.time()
print("For Loop Calc: {:.2f} in {:.2f} msecs".format(c,(t_end2-t_st2)*1000))


# use np inbuilt functions, such as:
np.exp(x)
np.abs(x)
np.maximum(0,x)
np.expand_dims(x,1)
np.log(x)
np.log2(x)
x**2
np.sum(x)

# implementing logistic regression with vectors
'''
# each iteration:

Z = wTX + b = np.dot(wT,X) + b
A = sigma(Z) = 1 / (1 + np.exp(Z))
dZ = A - Y
dW = (1/m)XdZT
dB = (1/m)np.sum(dZ)
W = W-alpha.dW
B = B-alpha.db
'''
i2 = np.array([np.random.randint(0,100,100),np.random.randint(0,100,100)])

# To transpose
r = np.random.randn(5,1)
r.shape
r.T.shape

a = np.random.randn(4, 3) # a.shape = (4, 3)
b = np.random.randn(3, 2) # b.shape = (3, 2)
np.dot(a,b)
c = a*b
c.shape

# notes
a * b # element-wise multiplication
np.dot(a,b) # matrix multiplication


