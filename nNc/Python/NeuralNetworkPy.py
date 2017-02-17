import numpy
#scipy.special for the sigmoid function expit()
import scipy.special
#neural network python implementation
#library for plotting arrays
import matplotlib.pyplot
#ensure teh losts are inside this notebook, not an external window
#%matplotlib inline 

class neuralNetwork:

    #initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        #set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
               
        #link weight matrices, wih and who
        #weights inside the arrays are w_i_j, where is from node to node j in the next layer
        #w11 w21
        #w12 w22 etc
        self.wih = numpy.random.normal(0.0, pow(self.hnodes,-0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes,-0.5), (self.onodes, self.hnodes))
        
        #learning rate
        self.lr = learningrate

        #activation function is the sigmoid function.
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # train the neural network
    def train(self, inputs_list, targets_list):
        #convert inputs list to 2d array T for transpose
        inputs = numpy.array(inputs_list, ndmin = 2).T
        targets = numpy.array(targets_list, ndmin =2).T

        #calculate signals into hidden layer //output decimal array 0.01< x<0.99
        hidden_inputs = numpy.dot(self.wih, inputs)
        #calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        #output layer error is the (target-actual)
        output_errors = targets - final_outputs
        #hidden layer error is the output_errors, split by weights, recombined at hidden nodes by backpropagation of error
        hidden_errors = numpy.dot(self.who.T, output_errors)

        #update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors*final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs))

        #update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors*hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        pass

    #query the result
    def query(self, inputs_list):
        #convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin = 2).T

        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        #calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    # backquery the neural network
    # we'll use the same termnimology to each item, 
    # eg target are the values at the right of the network, albeit used as input
    # eg hidden_output is the signal to the right of the middle nodes
    def backquery(self, targets_list):
        # transpose the targets list to a vertical array
        final_outputs = numpy.array(targets_list, ndmin=2).T
        
        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)

        # calculate the signal out of the hidden layer
        hidden_outputs = numpy.dot(self.who.T, final_inputs)
        # scale them back to 0.01 to .99
        hidden_outputs -= numpy.min(hidden_outputs)
        hidden_outputs /= numpy.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01
        
        # calculate the signal into the hideen layer
        hidden_inputs = self.inverse_activation_function(hidden_outputs)
        
        # calculate the signal out of the input layer
        inputs = numpy.dot(self.wih.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= numpy.min(inputs)
        inputs /= numpy.max(inputs)
        inputs *= 0.98
        inputs += 0.01
        
        return inputs


input_nodes =784
hidden_nodes = 200
output_nodes = 10

learning_rate = 0.1

n= neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

#load the mnist training ata CSV file into a list
training_data_file = open("mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#train the neural network
#epochs is the number of times the training data set is unsed for training

epochs = 5

for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        #split the record by the ',' commas
        all_values = record.split(',')
        #scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:])/255.0 *0.99) + 0.01
        #create the target output values (all 0.01, except the desired label with is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        #all values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

#load the mist test datat csv file into a list
test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

#test the neural network

#scorecard for how well the network performs , initially empty
scorecard = []

#go throught allthe records int he rtest data set
for record in test_data_list:
    #split the record by the ',' commas
    all_values = record.split(',')
    #correct answer is first value
    correct_label = int(all_values[0])
    #print(correct_label, "correct label")
    #scale and shift the inputs 
    inputs = (numpy.asfarray(all_values[1:])/ 255.0 * 0.99 )+ 0.01
    #query the netowrk
    outputs = n.query(inputs)
    #the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    #print(label, "network's answer\n")
    #append correct or incorrect to list
    if (label == correct_label):
        #network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        #network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass
    pass

#calculate the performance score, the fraction of correct answer
scored_array = numpy.asarray(scorecard)
print("tested", len(scored_array))
print("performance = " , scored_array.sum()/ scored_array.size)

