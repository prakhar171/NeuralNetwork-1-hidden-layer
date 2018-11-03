import math
import random
import matplotlib.pyplot as plt

'''Logic discussed with Preetha Datta and Dhruv Agarwal
Code written seperately'''

class Neural_Network():
	
	def __init__ (self, inputs, hidden, outputs, n):
		self.inputs = inputs
		self.hidden = hidden
		self.outputs = outputs
		self.n = n
		self.training_data = training_data
		self.testing_data = testing_data
	
	def randomised_weights(self):
		W = [None,]
		w = [None,]
		for i in range(self.outputs + 1):
			temp = []
			for j in range(self.hidden + 1):
				temp.append(random.uniform(-0.1,0.1))
			W.append(temp)
		for j in range (self.hidden +1):
			temp = []
			for k in range (self.inputs+1):
				temp.append(random.uniform(-0.1,0.1))
			w.append(temp)
		self.w = w
		self.W = W
		return w, W

	def f (self,s):
		ans = 1/(1 + math.exp(-10 * s))
		return ans

	def derivative(self,value):
		y = self.f(value)
		ans = y * (1 - y)
		return ans

	def f_pass(self, inputs_outputs, w, W, l):
		h = [1]
		s_j = [0]
		estimator = [None]
		s_i = [0]
		
		for j in range (1, self.hidden + 1):
			s = 0
			for k in range (self.inputs + 1):
				# print inputs_outputs[l][k]
				s += w[j][k] * inputs_outputs[l][k]
			s_j.append(s)
			h.append(self.f(s))

		for i in range (1, self.outputs+1):
			s = 0
			for j in range (self.hidden + 1):
				s += W[i][j] * h[j]
			s_i.append(s)
			estimator.append(self.f(s))

		return s_i, s_j, h, estimator

	def b_pass(self, s_i, s_j, estimator, W, l):
		del_i = [None]
		del_j = [None]

		for i in range(1, self.outputs+1):
			delta = (self.training_data[l][self.inputs+i] - estimator[i]) * self.derivative(s_i[i])
			del_i.append(delta)

		for j in range (1, self.hidden + 1):
			delta = None
			for i in range(1, self.outputs + 1):
				delta = del_i[i] * W[i][j]
			delta *= self.derivative(s_j[j])
			del_j.append(delta)

		return del_i, del_j

	def train(self):
		values_j = []
		x = []
		w, W = self.randomised_weights()
		J = 10
		epochs = 0
		while J > 0.01:
		# for i in range (40):
			J = 0
			for l in range (len(self.training_data)):
				s_i, s_j, h, estimator = self.f_pass(self.training_data,w,W,l)
				del_i, del_j = self.b_pass(s_i, s_j, estimator, W, l)
				for i in range (1, self.outputs+1):
					for j in range (1, self.hidden + 1):
						W[i][j] += self.n * del_i[i] * h[j]
				for j in range (1, self.hidden + 1):
					for k in range(1 + self.inputs):
						w[j][k] += self.n * del_j[j] * training_data[l][k]

			for l in range(len(self.training_data)):
				s_i, s_j, h, estimator = self.f_pass(self.training_data,w, W, l)
				for i in range(1, self.outputs + 1):
					J += (training_data[l][self.inputs+1] - estimator[i]) ** 2
			
			print J
			values_j.append(J)
			x.append(epochs)
			epochs += 1
		self.graph(x, values_j)

	def graph(self, axis1, axis2):
		fig, ax = plt.subplots(figsize = (10,6))
		ax.grid(color='black', linestyle='--', linewidth=0.1)
		plt.plot(axis1,axis2)
		plt.xlabel("Epochs")
		plt.ylabel("Error")
		plt.show()

	def test(self):
		w = self.w
		W = self.W

		sum_total = 0
		for l in range(len(self.testing_data)):
			# print l
			s_i, s_j, h, estimator = self.f_pass(self.testing_data,w, W, l)
			partial_sum = 0
			for i in range(1, self.outputs + 1):
				partial_sum += (training_data[l][self.inputs+1] - estimator[i]) ** 2

			sum_total += partial_sum
		
		return sum_total

def data (file):
	input_file_data=[]

	for i in file:
		value = i.strip('\r\n').split()
		input_file_data.append(value)
		
	for i in range(len(input_file_data)):
		for j in range(len(input_file_data[i])):
			input_file_data[i][j] = float(input_file_data[i][j])

	for i in range(len(input_file_data)):
		input_file_data[i].insert(0,1)

	return input_file_data

eeta = input("Enter Eeta: ")
hidden_neurons = input("Enter number of hidden neurons: ")

training = open("hw3trainingdata.txt", "r")
training_data = data(training)

testing = open("hw3testingdata.txt", "r")
testing_data = data(testing)

program = Neural_Network(1, hidden_neurons, 1, eeta)

# print tests
program.train()

x = program.test()
if x != None:
	print "Error in testing data is", x