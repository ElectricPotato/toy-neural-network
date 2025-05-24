

import numpy as np

class Network:
    def __init__(self, architecture, activation_func, randomise = True,
                 layer_values = None):
        # layer_values is for supplying your own values
        # if layer_values is not supplied, randomise sets all weights and biases
        # to a random value -1 to +1


        if layer_values is None:
            self.weights_list = []
            self.biases_list = []
            #for each layer, make a matrix for weights and a vector for biases
            for n_signals_in, n_signals_out in zip(architecture, architecture[1:]):
                #n_signals_in is the number of columns (width)
                #n_signals_out is the number of rows (height)
                #np.zeros takes (height, width) in that order
                weights_shape = (n_signals_out, n_signals_in)
                biases_shape = (n_signals_out, 1)

                if randomise:
                    weights = np.random.rand(*weights_shape) * 2 - 1
                    biases = np.random.rand(*biases_shape) * 2 - 1
                else:
                    weights = np.zeros(weights_shape)
                    biases = np.zeros(biases_shape)

                self.weights_list.append(weights)
                self.biases_list.append(biases)
        else:
            self.weights_list, self.biases_list = layer_values

        self.activation_func = np.vectorize(activation_func)
        
    def run(self, network_inputs):
        signals = network_inputs
        for weights, biases in zip(self.weights_list, self.biases_list):
            signals = self.activation_func(weights * signals + biases)
        return signals
    
    def show(self):
        for idx, (weights, biases) in enumerate(zip(self.weights_list, self.biases_list)):
            print(f"weights layer {idx}")
            print(weights)
            print(f"biases layer {idx}")
            print(biases)

def network_score(network, testcases):
    score = 0
    for testcase_input, testcase_expected_output in testcases:
        testcase_output = network.run(testcase_input)
        score -= np.sum(np.square(testcase_output - testcase_expected_output))
    return score

def network_results(network, testcases):
    outputs = []
    for testcase_input, testcase_expected_output in testcases:
        testcase_output = network.run(testcase_input)
        outputs.append(testcase_output)
    return outputs
    

def main():

    # "architecture" is the number of inputs, followed by the number of neurons
    # in each layer. The last layer is the output layer
    xor_architecture = [2, 2, 1]


    #https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Piecewise-linear_variants
    def leaky_relu(x):
        return x if x >= 0 else x * 0.1
        #equivalent to: max(x, 0.1 * x)

    xor_activation_func = leaky_relu

    # The XOR example is not that good of a general example because there are no
    # test cases that it can be run on that are outside of its training data

    # each testcase is a touple of a list of the input signals and a list of
    # the expected output signals
    testcases_XOR = [
        (np.matrix("0; 0"), np.matrix("0")),
        (np.matrix("0; 1"), np.matrix("1")),
        (np.matrix("1; 0"), np.matrix("1")),
        (np.matrix("1; 1"), np.matrix("0"))
    ]

    # generate some random networks and show the best one
    # TODO: add genetic algorithm or gradient descent

    print()
    print("1. Generating some random networks and scoring them")
    best_score = None
    best_network = None
    for i in range(10000):
        xor_network = Network(
            architecture=xor_architecture,
            activation_func=xor_activation_func,
            )
        
        score = network_score(xor_network, testcases_XOR)

        if (best_score is None) or (score > best_score):
            best_score = score
            best_network = xor_network

        if(i%1000==0):print(f"Best scoring network so far: {best_score}")

    print()
    print("Best scoring network found:")
    best_network.show()

    outputs = network_results(best_network, testcases_XOR)
    expected_outputs = [testcase_expected_output for testcase_input, testcase_expected_output in testcases_XOR]
    print("Best scoring network outputs:")
    print(outputs)
    print("Expected outputs:")
    print(expected_outputs)
    print(f"Score: {best_score}")

    #manually selected weights and biases
    #score is close to 0 (performs well)
    print()
    print("2. Scoring a network with manually selected weights")
    xor_network = Network(
        architecture=xor_architecture,
        activation_func=xor_activation_func,
        layer_values = (
            [np.matrix("1.0 -1.0; -1.0 1.0"), np.matrix("1.0 1.0")], #layer 0
            [np.matrix("0.0; 0.0"), np.matrix("0.0")] #layer 1
        )
    )
        
    score = network_score(xor_network, testcases_XOR)
    xor_network.show()

    outputs = network_results(xor_network, testcases_XOR)
    expected_outputs = [testcase_expected_output for testcase_input, testcase_expected_output in testcases_XOR]
    print("Best scoring network outputs:")
    print(outputs)
    print("Expected outputs:")
    print(expected_outputs)
    print(f"Score: {score}")
    

    

if __name__ == "__main__":
    main()
    #test1()