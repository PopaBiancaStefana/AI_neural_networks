import random
import sys
from numpy import exp, array, random, dot, around, sum

LEFT_BOUND = -0.5
RIGHT_BOUND = 0.5
E_MIN = 0.001


def read_input_from_file(filename):
    with open(filename) as file:
        epochs = int(next(file))
        learning_rate = float(next(file))
        output = array([[int(value) for value in next(file).split()]]).T

        return epochs, learning_rate, output


def sigmoid(x):
    return 1 / (1 + exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self):
        random.seed()

        input_layer_nodes = 2
        hidden_layer_nodes = 2

        self.synaptic_weights1 = ((RIGHT_BOUND - LEFT_BOUND) * random.rand(input_layer_nodes + 1,
                                                                           hidden_layer_nodes + 1)) + LEFT_BOUND

        self.synaptic_weights2 = ((RIGHT_BOUND - LEFT_BOUND) * random.rand(hidden_layer_nodes + 1, 1)) + LEFT_BOUND

    def train(self, training_inputs, training_outputs, epochs, learning_rate):
        done = False
        i = 0

        while not done:
            # forward propagation
            hidden_layer_output = sigmoid(dot(training_inputs, self.synaptic_weights1))
            network_output = sigmoid(dot(hidden_layer_output, self.synaptic_weights2))

            # output layer gradients and adjustments
            output_layer_error = training_outputs - network_output
            output_layer_gradient = output_layer_error * sigmoid_derivative(network_output)
            output_layer_adjustment = dot(learning_rate * hidden_layer_output.T, output_layer_gradient)
            self.synaptic_weights2 = self.synaptic_weights2 + output_layer_adjustment

            # hidden layer gradients and adjustments
            hidden_layer_error = dot(self.synaptic_weights2, output_layer_gradient.T)
            hidden_layer_gradient = hidden_layer_error * sigmoid_derivative(hidden_layer_output).T
            hidden_layer_adjustment = dot(learning_rate * training_inputs.T, hidden_layer_gradient.T)
            self.synaptic_weights1 = self.synaptic_weights1 + hidden_layer_adjustment

            mean_squared_error = sum(output_layer_error ** 2) / len(training_inputs)

            if i == epochs or mean_squared_error < E_MIN:
                done = True
            else:
                i += 1

    def forward_pass(self, inputs):
        hidden_layer_output = sigmoid(dot(inputs, self.synaptic_weights1))
        network_output = sigmoid(dot(hidden_layer_output, self.synaptic_weights2))
        return around(network_output, 3)


def main():
    input_data = array([[0, 0, -1], [0, 1, -1], [1, 0, -1], [1, 1, -1]])
    filename = sys.argv[1]
    epochs, learning_rate, output_data = read_input_from_file(filename)
    neural_network = NeuralNetwork()
    neural_network.train(input_data, output_data, epochs, learning_rate)

    print(neural_network.forward_pass(input_data))


if __name__ == '__main__':
    main()
