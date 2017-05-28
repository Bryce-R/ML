from mnist import read, show
import numpy as np
import os
import pdb
import math
import sys

def sigmoid(x):
    return 1.0/( 1.0 + np.exp(-x) )

def binary_classifier_acuracy(theta, X, y):
    y_hat_linear = np.dot(X, theta)
    y_hat = np.empty(list(y.shape))
    m = y.shape[0]
    for i in range(m):
        y_hat[i] =  sigmoid(y_hat_linear[i]) > 0.5
    correct = np.sum(y_hat == y)
    return float(correct)/m

def LogisticObjective(X, y, theta ):
    y_hat_linear = np.dot(train_x,theta)
    y_hat = np.empty(list(y.shape))
    m = y.shape[0]
    for i in range(m):
        y_hat[i] =  sigmoid(y_hat_linear[i])
    # pdb.set_trace()
    f = -np.sum( np.dot(train_y.T, np.log(y_hat)) + np.dot( 1.0-train_y.T, np.log(1.0-y_hat)) )
    return f

def GradCheck(X,y,theta, calc_grad):
    n = theta.shape[0]
    g_diff = np.empty( [n,1] )
    epsilon = 1e-4
    for i in range(n):
        print i,
        add = np.zeros( (n,1) )
        add[i] = epsilon
        g_diff[i] = (LogisticObjective(X,y,theta+add) - LogisticObjective(X,y,theta-add) )/(2*epsilon)
    print "checking gradient"
    for i in range(n):
        if math.fabs(calc_grad[i]-g_diff[i])>1e-4:
            pdb.set_trace()
            print ("Gradient checking failed. ")



def LogisticRegression(train_x,train_y,test_x,test_y):
    m,n = train_x.shape
    m_test = test_x.shape[0]
    alpha = 5e-5
    theta = np.random.rand(n,1)*0.001
    ct = 0
    maxIter = 200
    # Prev_train_obej = float('Inf')
    while (ct < maxIter):
        ct += 1
        # pdb.set_trace()
        y_hat_linear = np.dot(train_x,theta)
        y_hat = np.empty(list(train_y.shape))
        # sigmoid = lambda x: 1.0/(1.0 + math.exp(-x))
        for i in range(m):
            y_hat[i] =  sigmoid(y_hat_linear[i])
        train_obej = LogisticObjective(train_x, train_y, theta)
        if ct!=1 and math.fabs((train_obej-Prev_train_obej)/Prev_train_obej) < 1e-12: 
            print ("Converged. Iteration: {}. Training set objective function: {} . \n").format(ct, train_obej)
            break
        Prev_train_obej = train_obej
        # print ( 'Iteration {}: Training set objective function: {} \r'.format(ct, train_obej) )
        # sys.stdout.flush()
        # sys.stdout.write( 'Iteration {}: Train set objective function: {} \r'.format(ct, train_obej) )
        Gradient = np.dot(train_x.T, (y_hat - train_y) )
        # pdb.set_trace()
        # GradCheck(train_x,train_y, theta, Gradient)
        # pdb.set_trace()
        theta += -(alpha * Gradient)
        assert theta.shape == (n,1)
    # print binary_classifier_acuracy(theta, train_x, train_y)
    print ""
    print ("Training set accuracy: {0:0.2f}%".format(100*binary_classifier_acuracy(theta, train_x, train_y)) )
    print ("Testing set accuracy:  {0:0.2f}%".format(100*binary_classifier_acuracy(theta, test_x, test_y)  ) )








cwd = os.getcwd()
# "training" or "testing" 
train_images = read("training",cwd)

n = 28*28
m = 60000 #No of training sets, max: 60000
train_y = np.empty([m,1])
train_x = np.empty([m,n])
i = 0
for image in train_images:
    if image[0] == 0 or image[0] == 1:
        train_x[i,:] = np.reshape(image[1],(1,n))
        train_y[i] = image[0]
        i += 1
m = i
train_x = train_x[:i,:]
train_y = train_y[:i]
mean_x = np.mean(train_x, axis=0)
std_x  = np.std(train_x,  axis=0) + 0.1

train_x = (train_x - mean_x)/std_x
train_x = np.concatenate( (np.ones( (m,1) ),train_x), axis=1 )
# pdb.set_trace()
print 'No. of training examples: m = {}'.format(train_x.shape[0])
# testing images
m_test = 10000 #No of test sets, max: 10000
test_images = read("testing",cwd)
test_y = np.empty([m_test,1])
test_x = np.empty([m_test,n])
i = 0
for image in test_images:
    if image[0] == 0 or image[0] == 1:
        test_x[i,:] = np.reshape(image[1],(1,n))
        test_y[i] = image[0]
        i += 1
m_test = i
test_x = test_x[:i,:]
test_y = test_y[:i]
test_x = (test_x - mean_x)/std_x
test_x = np.concatenate( (np.ones( (m_test,1) ),test_x), axis=1 )
# pdb.set_trace()
print 'No. of testing examples: m = {}'.format(test_x.shape[0])
LogisticRegression(train_x,train_y,test_x,test_y)