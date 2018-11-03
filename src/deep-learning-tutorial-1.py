from keras.layers import Dense  # build our layers library
from keras.models import Sequential  # initialize neural network library
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import sys
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


data_path = sys.path[0] + '/../data/sign-language-digits-dataset'


'''
Overview the Data Set
- We will use "sign language digits data set" for this tutorial.
- In this data there are 2062 sign language digits images.
- As you know digits are from 0 to 9. Therefore there are 10 unique sign.
- At the beginning of tutorial we will use only sign 0 and 1 for simplicity.
- In data, sign zero is between indexes 204 and 408. Number of zero sign is 205.
- Also sign one is between indexes 822 and 1027. Number of one sign is 206.
  Therefore, we will use 205 samples from each classes(labels).
- Note: Actually 205 sample is very very very little for deep learning.
  But this is tutorial so it does not matter so much.
- Lets prepare our X and Y arrays.
  X is image array (zero and one signs) and Y is label array (0 and 1).
'''
# load data set
x_l = np.load(data_path + '/X.npy')
Y_l = np.load(data_path + '/Y.npy')
img_size = 64
plt.subplot(1, 2, 1)
plt.imshow(x_l[260].reshape(img_size, img_size))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(x_l[900].reshape(img_size, img_size))
plt.axis('off')
plt.show()


'''
- In order to create image array, I concatenate zero sign and one sign arrays
- Then I create label array 0 for zero sign images and 1 for one sign images.
'''
# Join a sequence of arrays along an row axis.
# from 0 to 204 is zero sign and from 205 to 410 is one sign
X = np.concatenate((x_l[204:409], x_l[822:1027]), axis=0)
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z, o), axis=0).reshape(X.shape[0], 1)
print("X shape: ", X.shape)
print("Y shape: ", Y.shape)
# X shape:  (410, 64, 64)
# Y shape:  (410, 1)

'''
- The shape of the X is (410, 64, 64)
  - 410 means that we have 410 images (zero and one signs)
  - 64 means that our image size is 64x64 (64x64 pixels)
- The shape of the Y is (410,1)
  - 410 means that we have 410 labels (0 and 1)
- Lets split X and Y into train and test sets.
  - test_size = percentage of test size. test = 15% and train = 75%
  - random_state = use same seed while randomizing.
    It means that if we call train_test_split repeatedly, it always creates same
    train and test distribution because we have same random_state.
'''
# Then lets create x_train, y_train, x_test, y_test arrays
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.15, random_state=42)
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]

'''
- Now we have 3 dimensional input array (X) so we need to make it flatten (2D)
  in order to use as input for our first deep learning model.
- Our label array (Y) is already flatten(2D) so we leave it like that.
- Lets flatten X array(images array).
'''
X_train_flatten = X_train.reshape(
    number_of_train, X_train.shape[1] * X_train.shape[2])
X_test_flatten = X_test .reshape(
    number_of_test, X_test.shape[1] * X_test.shape[2])
print("X train flatten", X_train_flatten.shape)
print("X test flatten", X_test_flatten.shape)
# X train flatten (348, 4096)
# X test flatten (62, 4096)

'''
- As you can see, we have 348 images and each image has 4096 pixels in image
  train array.
- Also, we have 62 images and each image has 4096 pixels in image test array.
- Then lets take transpose. You can say that WHYY, actually there is no
  technical answer. I just write the code(code that you will see oncoming parts)
  according to it :)
'''
x_train = X_train_flatten.T
x_test = X_test_flatten.T
y_train = Y_train.T
y_test = Y_test.T
print("x train: ", x_train.shape)
print("x test: ", x_test.shape)
print("y train: ", y_train.shape)
print("y test: ", y_test.shape)
# x train:  (4096, 348)
# x test:  (4096, 62)
# y train:  (1, 348)
# y test:  (1, 62)

'''
What we did up to this point:
- Choose our labels (classes) that are sign zero and sign one
- Create and flatten train and test sets
'''


'''
Logistic Regression
- When we talk about binary classification (0 and 1 outputs) what comes to mind
  first is logistic regression.
- However, in deep learning tutorial what to do with logistic regression there??
- The answer is that logistic regression is actually a very simple neural
  network.
- By the way neural network and deep learning are same thing. When we will come
  artificial neural network, I will explain detailed the terms like "deep".
- In order to understand logistic regression (simple deep learning) lets first
  learn computation graph.
'''

'''
Initializing parameters
- As you know input is our images that has 4096 pixels(each image in x_train).
- Each pixels have own weights.
- The first step is multiplying each pixels with their own weights.
- The question is that what is the initial value of weights?
  - There are some techniques that I will explain at artificial neural network
    but for this time initial weights are 0.01.
  - Okey, weights are 0.01 but what is the weight array shape? As you understand
    from computation graph of logistic regression, it is (4096,1)
  - Also initial bias is 0.
- Lets write some code. In order to use at coming topics like artificial neural
  network (ANN), I make definition (method).
'''


def initialize_weights_and_bias(dimension):
    # lets initialize parameters
    # So what we need is dimension 4096 that is number of pixels as a parameter
    # for our initialize method(def)
    w = np.full((dimension, 1), 0.01)
    b = 0.0
    return w, b


w, b = initialize_weights_and_bias(4096)

'''
Forward Propagation
- The all steps from pixels to cost is called forward propagation
  - z = (w.T)x + b => in this equation we know x that is pixel array, we know
    w (weights) and b (bias) so the rest is calculation. (T is transpose)
  - Then we put z into sigmoid function that returns y_head(probability).
    When your mind is confused go and look at computation graph.
    Also equation of sigmoid function is in computation graph.
  - Then we calculate loss(error) function.
  - Cost function is summation of all loss(error).
  - Lets start with z and the write sigmoid definition(method) that takes z as
    input parameter and returns y_head(probability)
'''


# calculation of z
# z = np.dot(w.T,x_train)+b


def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head


y_head = sigmoid(0)
print(y_head)
# 0.5


def forward_propagation(w, b, x_train, y_train):
    # Forward propagation steps:
    # find z = w.T*x+b
    # y_head = sigmoid(z)
    # loss(error) = loss(y,y_head)
    # cost = sum(loss)
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)  # probabilistic 0-1
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    # x_train.shape[1]  is for scaling
    cost = (np.sum(loss))/x_train.shape[1]
    return cost


def forward_backward_propagation(w, b, x_train, y_train):
    # In backward propagation we will use y_head that found in forward progation
    # Therefore instead of writing backward propagation method, lets combine forward propagation and backward propagation

    # forward propagation
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    # x_train.shape[1]  is for scaling
    cost = (np.sum(loss))/x_train.shape[1]
    # backward propagation
    # x_train.shape[1]  is for scaling
    derivative_weight = (
        np.dot(x_train, ((y_head-y_train).T)))/x_train.shape[1]
    # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    gradients = {"derivative_weight": derivative_weight,
                 "derivative_bias": derivative_bias}
    return cost, gradients


def update(w, b, x_train, y_train, learning_rate, number_of_iterarion):
    # Updating(learning) parameters
    cost_list = []
    cost_list2 = []
    index = []
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iterarion):
        # make forward and backward propagation and find cost and gradients
        cost, gradients = forward_backward_propagation(w, b, x_train, y_train)
        cost_list.append(cost)
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration %i: %f" % (i, cost))
    # we update(learn) parameters weights and bias
    parameters = {"weight": w, "bias": b}
    plt.plot(index, cost_list2)
    plt.xticks(index, rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list
# parameters, gradients, cost_list = update(
#     w, b, x_train, y_train, learning_rate = 0.009,number_of_iterarion = 200)


def predict(w, b, x_test):
    # prediction

    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T, x_test)+b)
    Y_prediction = np.zeros((1, x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(z.shape[1]):
        if z[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    return Y_prediction
# predict(parameters["weight"],parameters["bias"],x_test)


def logistic_regression(x_train, y_train, x_test, y_test, learning_rate,  num_iterations):
    # initialize
    dimension = x_train.shape[0]  # that is 4096
    w, b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(
        w, b, x_train, y_train, learning_rate, num_iterations)

    y_prediction_test = predict(
        parameters["weight"], parameters["bias"], x_test)
    y_prediction_train = predict(
        parameters["weight"], parameters["bias"], x_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(
        100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(
        100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))


# execute custom logistic regression
logistic_regression(x_train, y_train, x_test, y_test,
                    learning_rate=0.01, num_iterations=150)

# execute sklearn logistic regression
logreg = linear_model.LogisticRegression(
    random_state=42, max_iter=150, solver='liblinear', multi_class='auto')
print("test accuracy: {} ".format(logreg.fit(
    x_train.T, y_train.T.ravel()).score(x_test.T, y_test.T)))
print("train accuracy: {} ".format(logreg.fit(
    x_train.T, y_train.T.ravel()).score(x_train.T, y_train.T)))


'''
2-Layer Neural Network
'''


def initialize_parameters_and_layer_sizes_NN(x_train, y_train):
    # intialize parameters and layer sizes
    parameters = {"weight1": np.random.randn(3, x_train.shape[0]) * 0.1,
                  "bias1": np.zeros((3, 1)),
                  "weight2": np.random.randn(y_train.shape[0], 3) * 0.1,
                  "bias2": np.zeros((y_train.shape[0], 1))}
    return parameters


def forward_propagation_NN(x_train, parameters):

    Z1 = np.dot(parameters["weight1"], x_train) + parameters["bias1"]
    A1 = np.tanh(Z1)
    Z2 = np.dot(parameters["weight2"], A1) + parameters["bias2"]
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


def compute_cost_NN(A2, Y, parameters):
    # Compute cost
    logprobs = np.multiply(np.log(A2), Y)
    cost = -np.sum(logprobs)/Y.shape[1]
    return cost


def backward_propagation_NN(parameters, cache, X, Y):
    # Backward Propagation
    dZ2 = cache["A2"]-Y
    dW2 = np.dot(dZ2, cache["A1"].T)/X.shape[1]
    db2 = np.sum(dZ2, axis=1, keepdims=True)/X.shape[1]
    dZ1 = np.dot(parameters["weight2"].T, dZ2)*(1 - np.power(cache["A1"], 2))
    dW1 = np.dot(dZ1, X.T)/X.shape[1]
    db1 = np.sum(dZ1, axis=1, keepdims=True)/X.shape[1]
    grads = {"dweight1": dW1,
             "dbias1": db1,
             "dweight2": dW2,
             "dbias2": db2}
    return grads


def update_parameters_NN(parameters, grads, learning_rate=0.01):
    # update parameters
    parameters = {
        "weight1": parameters["weight1"]-learning_rate*grads["dweight1"],
        "bias1": parameters["bias1"]-learning_rate*grads["dbias1"],
        "weight2": parameters["weight2"]-learning_rate*grads["dweight2"],
        "bias2": parameters["bias2"]-learning_rate*grads["dbias2"]
    }
    return parameters


def predict_NN(parameters, x_test):
    # prediction

    # x_test is a input for forward propagation
    A2, cache = forward_propagation_NN(x_test, parameters)
    Y_prediction = np.zeros((1, x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(A2.shape[1]):
        if A2[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    return Y_prediction


def two_layer_neural_network(x_train, y_train, x_test, y_test, num_iterations):
    # 2 - Layer neural network
    cost_list = []
    index_list = []
    # initialize parameters and layer sizes
    parameters = initialize_parameters_and_layer_sizes_NN(x_train, y_train)

    for i in range(0, num_iterations):
         # forward propagation
        A2, cache = forward_propagation_NN(x_train, parameters)
        # compute cost
        cost = compute_cost_NN(A2, y_train, parameters)
        # backward propagation
        grads = backward_propagation_NN(parameters, cache, x_train, y_train)
        # update parameters
        parameters = update_parameters_NN(parameters, grads)

        if i % 100 == 0:
            cost_list.append(cost)
            index_list.append(i)
            print("Cost after iteration %i: %f" % (i, cost))
    plt.plot(index_list, cost_list)
    plt.xticks(index_list, rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()

    # predict
    y_prediction_test = predict_NN(parameters, x_test)
    y_prediction_train = predict_NN(parameters, x_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(
        100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(
        100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    return parameters


parameters = two_layer_neural_network(
    x_train, y_train, x_test, y_test, num_iterations=2500)


'''
L Layer Neural Network
'''
# reshaping
x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T

# Evaluating the ANN


def build_classifier():
    classifier = Sequential()  # initialize neural network
    classifier.add(Dense(units=8, kernel_initializer='uniform',
                         activation='relu', input_dim=x_train.shape[1]))
    classifier.add(
        Dense(units=4, kernel_initializer='uniform', activation='relu'))
    classifier.add(
        Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


classifier = KerasClassifier(build_fn=build_classifier, epochs=100)
accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=3)
mean = accuracies.mean()
variance = accuracies.std()
print("Accuracy mean: " + str(mean))
print("Accuracy variance: " + str(variance))
