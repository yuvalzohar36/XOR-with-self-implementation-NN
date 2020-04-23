import numpy as np
import random
import gzip
from sklearn.utils import shuffle


class NeuralNetwork:
    def __init__(self, layers):
        '''
        create biases and weights arrays.
        biases array - > create bias for each node (neuron) in the second layer and forword (including the output neuron)
        weights array - > create weight for each node from all the neurons in the previous layer.
        every node will get an array with random values, and the size will be the node's quantity in the previous layer.
        '''
        self.num_layers = len(layers)
        self.layers = layers
        self.biases, self.weights, self.activations, self.Z_list = [], [], [], []
        self.learning_rate = 0.05
        for i in range(1, len(layers)):
            self.biases.append(np.random.rand(self.layers[i], 1))
        for i in range(len(layers) - 1):
            self.weights.append(np.random.rand(self.layers[i + 1], self.layers[i]))

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def derivative_sigmoid(x):
        """Derivative of the sigmoid function."""
        return NeuralNetwork.sigmoid(x) * (1 - NeuralNetwork.sigmoid(x))

    def feed_forword(self, input):
        '''
        feed_forward function ----- >
        get input parameter (an array), take the first raw in the self.weights[i] and the inputs, dot them, and add the biases[i],
        then sigmoid all of that.
        all this operation equals to output, and then feed the next layer with the output of the previous layer,
        with the same operation.
        the input should be array (n , 1) shape. for example : [[1],[2],[3]]
        '''
        output = input
        for i in range(len(self.biases)):
            output = self.sigmoid(np.dot(self.weights[i], output) + self.biases[i])
        return output

    def train(self, inputs, targets, *epochs_args):
        '''
        Z_list ----> sigmoid prime(Weights * inputs + biases).
        activations ----> sigmoid function(Weights * inputs + biases)
        outputs ----> the output, use for the cost function to calculate the error.
        errors ----> the the outputs and the target and calculate the error with the cost function.
        epoch ----> number of training, default = 1000.
        '''
        if not epochs_args: epochs = 1000
        else: epochs = epochs_args[0]
        for epoch in range(epochs):
            Z_list = []
            activations = [inputs]
            list_inputs = inputs
            for i in range(len(self.biases)):
                Z_list.append(NeuralNetwork.derivative_sigmoid(np.dot(self.weights[i], list_inputs) + self.biases[i]))
                list_inputs = self.sigmoid(np.dot(self.weights[i], list_inputs) + self.biases[i])
                activations.append(list_inputs)
            outputs = self.feed_forword(inputs)
            errors = self.cost(outputs, targets)
            #new_biases, new_weights ----> will provide the new values for self.biases, self.weights.
            new_biases = [np.zeros(b.shape) for b in self.biases]
            new_weights = [np.zeros(w.shape) for w in self.weights]
            # Calculate cost derivative - #1
            cost_derivative = 2 * errors
            '''
            Back propogation ---->
            first time calculate the delta for the last layer,
            only the first layer multiplying with the cost_derivative.
            the iterate all the layers and calculate the delta with the the previous delta from the previous layer.
            every iterate add all the result to the new_biases, new weights.
            then reduce the (new_weights, new_biases) * self.learning_rate from self.weights, self.biases,
            for tuning all the weights and biases.
            '''
            delta = cost_derivative * Z_list[-1] * self.learning_rate
            new_biases[-1] = delta
            new_weights[-1] = np.dot(delta, activations[-2].transpose())

            for layer in range(2, self.num_layers):
                delta = np.dot(self.weights[-layer +1].transpose(), delta) * Z_list[-layer] * self.learning_rate
                new_biases[-layer] = delta
                new_weights[-layer] = np.dot(delta, np.transpose(activations[-layer-1]))

            for i in range(len(self.weights)):
                self.weights[i] -= new_weights[i] * self.learning_rate
            for j in range(len(self.biases)):
                self.biases[j] -= new_biases[j] * self.learning_rate

    def cost(self, outputs, desired_outputs):
        #this function calculate the cost of the function
        return (outputs - desired_outputs)

if __name__ == '__main__':
        nn = NeuralNetwork([2,9,1])
        input = [[[0],[0]],[[0],[1]],[[1],[0]],[[1],[1]]]
        target = [[0],[1],[1],[0]]
        for i in range(100000):
            inputs_shuffled, targets_shuffled = shuffle(np.array(input), np.array(target))
            for k in range(4):
                nn.train(inputs_shuffled[k], targets_shuffled[k], 100000)
        print(nn.feed_forword(input[0]))
        print(nn.feed_forword(input[1]))
        print(nn.feed_forword(input[2]))
        print(nn.feed_forword(input[3]))
