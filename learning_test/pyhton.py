#Neural Networks from Scratch in Python

inputs = [1, 2, 3, 2.5]

weights1 = [0.2, 0.8, -0.5, 1.0 ]
weights2 = [0.5, -0.91, 0.26, -0.5 ]
weights3 = [-0.26, -0.27, 0.17, 0.87 ]
bias1 = 2
bias2 = 3
bias3 = 0.5
output = [inputs[0] * weights1[0] + inputs[1] * weights1[1] + inputs[2] * weights1[2] + inputs[3] * weights1[3] + bias1,
		  inputs[0] * weights2[0] + inputs[1] * weights2[1] + inputs[2] * weights2[2] + inputs[3] * weights2[3] + bias2,
		  inputs[0] * weights3[0] + inputs[1] * weights3[1] + inputs[2] * weights3[2] + inputs[3] * weights3[3] + bias3 ]
print(output)


#for loop version
inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0 ],
			[0.5, -0.91, 0.26, -0.5 ],
			[-0.26, -0.27, 0.17, 0.87 ]]
biases = [2, 3, 0.5]

output_layer = []

for neuron_weigths, bias in zip (weights, biases):
	neuron_output = 0
	for n_input, n_weight in zip(inputs, neuron_weigths):
		neuron_output += n_input * n_weight
	neuron_output = neuron_output + bias
	output_layer.append(neuron_output)
	
print(output_layer)


#numpy version 
import numpy as np

inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0 ],
			[0.5, -0.91, 0.26, -0.5 ],
			[-0.26, -0.27, 0.17, 0.87 ]]

biases = [2, 3, 0.5]

output = np.dot(weights, inputs) + biases

print(output)


#more batches (transpose technique)
inputs =  [[1, 2, 3, 2.5],
		   [2.0, 5.0 , -1.0 , 2.0],
		   [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0 ],
			[0.5, -0.91, 0.26, -0.5 ],
			[-0.26, -0.27, 0.17, 0.87 ]]

biases = [2, 3, 0.5]

output = np.dot(inputs, np.array(weights).T) + biases

print(output)

