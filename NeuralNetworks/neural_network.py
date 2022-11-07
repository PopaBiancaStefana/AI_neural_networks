import math
import random

import pandas as pd
from numpy import exp, zeros, multiply, dot, array

input_layer_size = 0
hidden_layer_size = 0
output_layer_size = 0
learning_rate = 0
max_num_of_ages = 0
weights1 = []
weights2 = []
train_data_matrix = []


def read_data():
    global train_data_matrix

    df = pd.read_csv('dataset/iris.data')
    df = df.sample(frac=1)  # shuffle
    split_factor = 0.9
    n_train = math.floor(split_factor * df.shape[0])
    n_test = math.ceil((1 - split_factor) * df.shape[0])
    train_data = df.head(n_train)
    test_data = df.tail(n_test)
    print(train_data.to_string())
    print(test_data.to_string())

    train_data_matrix = train_data[["sepal_len", "sepal_width", "petal_len", "petal_width"]].to_numpy()


def initialize_parameters():
    global input_layer_size
    global hidden_layer_size
    global output_layer_size
    global learning_rate
    global max_num_of_ages
    global weights1
    global weights2

    input_layer_size = 4
    hidden_layer_size = 4
    output_layer_size = 3
    learning_rate = 0.1
    max_num_of_ages = 50

    # matrix for weights between input and hidden layer
    weights1 = zeros((input_layer_size, hidden_layer_size))
    # matrix for weights between hidden and output layer
    weights2 = zeros((hidden_layer_size, output_layer_size))

    for i in range(0, input_layer_size):
        for j in range(0, hidden_layer_size):
            weights1[i][j] = random.uniform(0, 1)

    for i in range(0, hidden_layer_size):
        for j in range(0, output_layer_size):
            weights2[i][j] = random.uniform(0, 1)

    print("weights1:", weights1)
    print("weights2:", weights2)


def sigmoid(x):
    return 1 / (1 + exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def forward_propagation():
    # forward propagation
    hidden_layer_output = sigmoid(array(train_data_matrix).reshape(len(train_data_matrix), 4).dot(weights1))
    network_output = sigmoid(dot(hidden_layer_output, weights2))

    print('Network output:', network_output)
    return hidden_layer_output



