# Chandrashekar Shetty - @iamDshetty
# NeuralNetwork - Deep Feed Forward
# Back propogation algorithm

#----NN Structure----
# input_nodes = 16
# hidden_nodes = 8
# output_nodes = 2


import numpy
import scipy.special

import matplotlib.pyplot

class neuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes)) # random array from o to 1/sqroot(totalinputs) withe shape of hnodes and inodes
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        print("weights")


        # learning rate
        self.lr = learningrate

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))



        pass

    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

input_nodes = 16
hidden_nodes = 8
output_nodes = 2

# learning rate
learning_rate = 0.2

g_input = (numpy.zeros(16) + 0.01)
g_input[5] = 0.99
g_input[9] = 0.99

g_output = numpy.zeros(2)+0.01
g_output[0]= 0.01
g_output[1]=0.99

n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

training_data_file = open("mnist_dataset/mytrainset.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
#print(n.wih)
for y in range(10000):
    for x in training_data_list:
        all_values = x.split(',')
        correct_label = int(all_values[0])
        inputs = numpy.asfarray(all_values[1:])
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs,targets)
        pass
    pass
#print(n.who)
#print("weights",n.wih)

test_data_file = open("mnist_dataset/mytestset.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

test_values = test_data_list[1].split(',')
test_input = numpy.asfarray(test_values)
#print(test_input)
print(n.query(test_input))