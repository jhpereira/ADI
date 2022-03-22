import numpy as np 
from fractions import Fraction
from collections import Counter
import matplotlib.pyplot as plt

transitionHW = np.array ([[0, 1/3, 1/3, 1/3], [1/2, 0, 0, 1/2], [1/2, 0, 0, 1/2], [1/3, 1/3, 1/3, 0]]) #From Homework
transitionLAB = np.load('pacman-big.npy') #From Lab

transition31 = np.array ([[0.2, 0.6, 0.15, 0.05], [0.2, 0.6, 0.15, 0.05], [0.05, 0.15, 0.6, 0.2], [0.05, 0.15, 0.6, 0.2]]) #Problem 3.1.
obs31 = np.array ([[1,0],[0,1],[1,0],[0,1]]) #Observation problem 3.1.
mu31 = np.array ([0.125, 0.375, 0.375, 0.125]) #Initial state dist problem 3.1.

transition9 = np.array([[0.25, 0.75, 0], [0, 0.25, 0.75], [0, 0, 1]]) #Problem 9, exercises
obs9 = np.array([[0.1, 0.9], [0.7, 0.3], [0.1, 0.9]]) #Observation problem 9
mu9 = np.array([1/3, 1/3, 1/3]) #Initial state dist, problem 9

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

def forward_algorithm(P_matrix, O_matrix, mu0, obs): #Solves example 3.1 from class notes, returns last alpha (normalized), for most likely state

	alpha = [0] * (len(obs))

	alpha[0] = (np.dot(np.diag(O_matrix[:,obs[0]]), mu0.T))

	for i in range (1,len(obs)):

		alpha[i] = np.dot(np.dot(np.diag(O_matrix[:,obs[i]]), P_matrix.T), alpha[i-1])

	forw = alpha[-1]

	normalize_factor = np.dot(forw.T, np.ones(forw.shape[0]))

	return (forw/normalize_factor)

def viterbi(P_matrix, O_matrix, mu0, obs): #Viterbi algorithm (uses forward_algo), would be better to merge both, dont feel doing it tho

	alpha = [0] * (len(obs))

	imax = [0] * (len(obs) - 1)

	alpha[0] = (np.dot(np.diag(O_matrix[:,obs[0]]), mu0.T))

	for i in range (1,len(obs)):

		aux = np.dot(P_matrix.T, np.diag(alpha[i-1]))

		imax[i-1] = np.argmax(aux, axis = 1)

		#imax[i-1] += np.ones((1,imax[i-1].shape[1]))

		aux = np.amax(aux, axis = 1)

		alpha[i] = np.dot(np.diag(O_matrix[:,obs[i]]), aux)

	backtrack = [0] * len(obs)

	backtrack[-1] = np.argmax(alpha[-1], axis = 0)

	for i in range (2, len(obs) + 1):
		
		backtrack[-i] = (imax[-i + 1])[backtrack[-i+1]]

	#backtrack += np.ones((1,len(obs))) #If states numbered 1 to N

	return (backtrack)

#markov_chain_monte_carlo(1, transitionLAB, 1000) #MCMC

#print(trans_limit(transitionLAB)[1])

#print(Fraction(compute_probability(17, 16, 1, transitionLAB)).limit_denominator(100)) #Probability of x1 = 16 | x0 = 17, should be 1/3

#print(Fraction(compute_probability(35, 29, 1, transitionLAB)).limit_denominator(100)) #Probability of x1 = 29 | x0 = 35, should be 1/2

#print(Fraction(compute_probability(35, 24, 4, transitionLAB)).limit_denominator(100)) #Probability of x4 = 24 | x0 = 35, should be 1/(2^4) = 1/16

print(forward_algorithm(transition31, obs31, mu31, (0,0,1)), 'Forward 3.1') #3.1. is right as per solutions

print(viterbi(transition31, obs31, mu31, (0,0,1)), 'Viterbi 3.1')

print(forward_algorithm(transition9, obs9, mu9, [1,0,1]), 'Forward 3.1') #9 is the same as my solution (by hand)

print(viterbi(transition9, obs9, mu9, (1,0,1)), 'Viterbi 9')



