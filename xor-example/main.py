

import numpy as np

import random

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
    
def print_summary(network, testcases):
    print("Best scoring network found:")
    network.show()

    outputs = network_results(network, testcases)
    score = network_score(network, testcases)
    expected_outputs = [testcase_expected_output for testcase_input, testcase_expected_output in testcases]
    print("Best scoring network outputs:")
    print(outputs)
    print("Expected outputs:")
    print(expected_outputs)
    print(f"Score: {score}")

def main():

    # "architecture" is the number of inputs, followed by the number of neurons
    # in each layer. The last layer is the output layer
    xor_architecture = [2, 2, 1]
    # 2 inputs,
    # intermediate layer with 2 neurons,
    # output layer with 1 neuron


    #https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Piecewise-linear_variants
    def leaky_relu(x):
        return x if x >= 0 else x * 0.1
        #equivalent to: max(x, 0.1 * x)

    xor_activation_func = leaky_relu


    # The simplest possible network thats still slightly interesting is an XOR gate
    # however, its not that good of a general example because there are no
    # test cases that it can be run on that are outside of its training data

    # each testcase is a touple of a list of the input signals and a list of
    # the expected output signals
    testcases_XOR = [
        (np.matrix("0; 0"), np.matrix("0")),
        (np.matrix("0; 1"), np.matrix("1")),
        (np.matrix("1; 0"), np.matrix("1")),
        (np.matrix("1; 1"), np.matrix("0"))
    ]

    def manually_selected():
        #manually selected weights and biases
        #score is close to 0 (performs well)
        print()
        print("1. Scoring a network with manually selected weights")
        xor_network = Network(
            architecture=xor_architecture,
            activation_func=xor_activation_func,
            layer_values = (
                [np.matrix("1.0 -1.0; -1.0 1.0"), np.matrix("1.0 1.0")], #layer 0
                [np.matrix("0.0; 0.0"), np.matrix("0.0")] #layer 1
            )
        )
        print_summary(xor_network, testcases_XOR)

    def generate_random():
        # generate some random networks and show the best one
        print()
        print("2. Generating some random networks and scoring them")
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
        print_summary(best_network, testcases_XOR)

    def genetic_algorithm():
        # use a genetic algorithm to 'crossbreed' solutions that score well
        print()
        print("3. Using a genetic algorithm to 'crossbreed' solutions that score well")

        n_generations = 20
        population_size = 20 #number of networks in each generation
        surviving_population_size = 8 #number of networks that are carried over into the next generation

        #create initial population
        population = []
        for i in range(10000):
            network = Network(
                architecture=xor_architecture,
                activation_func=xor_activation_func,
            )
            score = network_score(network, testcases_XOR)

            population.append((score, network))

        for generation in range(n_generations):
            #select the top performing networks
            population.sort(key = lambda x: x[0])
            surviving_population = population[:surviving_population_size]

            #generate the next generation of networks
            new_population = []
            for i in range(population_size):
                parent_a, parent_b = random.sample(surviving_population, 2) # select two random networks
                
                #TODO: crossbreed
                zip(parent_a.weights_list, parent_b.weights_list)
                zip(parent_a.biases_list, parent_b.biases_list)

                #TODO: mutate

                network = Network(
                    architecture=xor_architecture,
                    activation_func=xor_activation_func,
                )
                
                

        #TODO

    def gradient_descent():
        pass

        #TODO



    #manually_selected()

    #random is boring, not very efficient
    #generate_random()

    #genetic algorithm is still not very efficient, but a bit more interesting
    genetic_algorithm()

    #gradient descent is a bit more efficient
    #gradient_descent()
    

    

if __name__ == "__main__":
    main()