import numpy as np 
from fractions import Fraction
from collections import Counter
import matplotlib.pyplot as plt

transitionHW = np.array ([[0, 1/3, 1/3, 1/3], [1/2, 0, 0, 1/2], [1/2, 0, 0, 1/2], [1/3, 1/3, 1/3, 0]]) #From Homework

transitionLAB = np.load('pacman-big.npy') #From Lab

def get_next_state(state,transition): #Randomly gets next_state (based on current and transition prob.)#
	values = np.arange(1,transition.shape[0]+1)

	next_state = np.random.choice(values, p = transition[state-1])

	return (next_state) 

def markov_chain_monte_carlo(state, transition, iterations): #Steps to return (based on MC analysis)

	time_return = []

	for i in range (0,iterations):

		state0 = state
		time_step = 0

		statet = get_next_state(state0,transition)
		time_step += 1

		while statet != 1:
			statet = get_next_state(statet,transition)
			time_step += 1
			

		time_return.append(time_step)

	result = Counter(time_return)

	x_axis = result.keys()
	y_axis = result.values()

	x_axis_val = np.array(list(x_axis))
	y_axis_val = np.array(list(y_axis))

	avg = np.dot(x_axis_val, y_axis_val) / iterations

	print('Average time steps to return:', avg)

	plt.bar(x_axis, y_axis)

	plt.title('Number of simulations with x time steps to return to 1')

	plt.xlabel('Steps to return')

	plt.ylabel('Number of simulations')

	plt.show()

	return

def compute_probability(initial_state, final_state, time_steps, transition): #Returns probability of going to state x to y in k time_steps
	stateI = np.zeros([1,transition.shape[0]])
	stateI[0][initial_state-1] = 1 

	Ttrans = np.linalg.matrix_power(transition, time_steps)

	final_dist = np.dot(stateI, Ttrans)

	stateF = np.zeros([1,transition.shape[0]])
	stateF[0][final_state-1] = 1

	return (np.dot(final_dist,stateF.T)[0][0]) 

def trans_limit(transition): #Limit of matrix^n, not very good right now, just obtained large power
	
	return(np.linalg.matrix_power(transition,10000))

#markov_chain_monte_carlo(1, transitionLAB, 1000) #MCMC

#print(trans_limit(transitionLAB)[1])

print(Fraction(compute_probability(17, 16, 1, transitionLAB)).limit_denominator(100)) #Probability of x1 = 16 | x0 = 17, should be 1/3

print(Fraction(compute_probability(35, 29, 1, transitionLAB)).limit_denominator(100)) #Probability of x1 = 29 | x0 = 35, should be 1/2

print(Fraction(compute_probability(35, 24, 4, transitionLAB)).limit_denominator(100)) #Probability of x4 = 24 | x0 = 35, should be 1/(2^4) = 1/16



