from mnist import read, show
import numpy as np
import os
import pdb

def sigmoid(x):
    return 1.0/( 1.0 + np.exp(-x) )

def softmax_classifier_acuracy(theta, X, y):
    m,n = X.shape
    y_hat_linear = np.dot(X, theta)
    y_hat_exp = np.exp(y_hat_linear)
    y_hat = np.argmax(y_hat_exp, axis =1)
    # y_hat_exp_sum = np.sum(y_hat_exp, axis = 1)
    # y_hat = y_hat_exp/y_hat_exp_sum
    # pdb.set_trace()
    correct = np.sum(y_hat == y.T)
    return float(correct)/m

def softmaxObjective(X, y, theta):
    m,n = X.shape
    y_hat_linear = np.dot(X, theta)
    y_hat_exp = np.exp(y_hat_linear)
    y_hat_exp_sum = np.sum(y_hat_exp, axis = 1)
    f = 0.0
    for i in xrange(m):
        f += np.log( y_hat_exp[i , int(y[i])]/y_hat_exp_sum[i] )
    return f

def softmaxRegression(train_x,train_y,test_x,test_y):
    m,n = train_x.shape
    m_test = test_x.shape[0]
    theta = np.random.rand(n,10)*0.001
    alpha = 1e-5
    ct = 0
    maxIter = 100
    # Prev_train_obej = float('Inf')
    while (ct < maxIter):
        y_hat_linear = np.dot(train_x, theta)
        y_hat_exp = np.exp(y_hat_linear)
        y_hat_exp_sum = np.sum(y_hat_exp, axis = 1)
        gradient = np.zeros( (n,10) )
        for k in xrange(10):
            for i in xrange(m):
                # pdb.set_trace()
                gradient[:,k] -= ( (train_y[i] == k) - y_hat_exp[i,k]/y_hat_exp_sum[i] )*train_x[i,:].T
        train_obj = softmaxObjective(train_x,train_y,theta)
        print ( 'Iteration {}: Training set objective function: {} \r'.format(ct, train_obj) )
        theta += -(alpha * gradient)
        assert theta.shape == (n,10)
        ct += 1
    print ""
    print ("Training set accuracy: {0:0.2f}%".format(100*softmax_classifier_acuracy(theta, train_x, train_y)) )
    print ("Testing set accuracy:  {0:0.2f}%".format(100*softmax_classifier_acuracy(theta, test_x, test_y)  ) )

cwd = os.getcwd()
# "training" or "testing" 
train_images = read("training",cwd)

n = 28*28
m = 1000 #No of training sets, max: 60000
train_y = np.empty([m,1])
train_x = np.empty([m,n])
i = 0
for image in train_images:
    if i >= m: break
    train_x[i,:] = np.reshape(image[1],(1,n))
    train_y[i] = image[0]
    i += 1
train_x = train_x[:i,:]
train_y = train_y[:i]
mean_x = np.mean(train_x, axis=0)
std_x  = np.std(train_x,  axis=0) + 0.1
# pdb.set_trace()
train_x = (train_x - mean_x)/std_x
train_x = np.concatenate( (np.ones( (m,1) ),train_x), axis=1 )
print 'No. of training examples: m = {}'.format(train_x.shape[0])
# testing images
m_test = 100 #No of test sets, max: 10000
test_images = read("testing",cwd)
test_y = np.empty([m_test,1])
test_x = np.empty([m_test,n])
i = 0
for image in test_images:
    if i >= m_test: break
    test_x[i,:] = np.reshape(image[1],(1,n))
    test_y[i] = image[0]
    i += 1
m_test = i
test_x = test_x[:i,:]
test_y = test_y[:i]
test_x = (test_x - mean_x)/std_x
test_x = np.concatenate( (np.ones( (m_test,1) ),test_x), axis=1 )
print 'No. of testing examples: m = {}'.format(test_x.shape[0])
softmaxRegression(train_x,train_y,test_x,test_y)
    