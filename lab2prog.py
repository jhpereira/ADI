import numpy as np 
from fractions import Fraction
from collections import Counter
import matplotlib.pyplot as plt


def Value_Iteration(cost, probabilities, gamma):

	J0 = np.zeros((np.shape(cost)[0],1))

	for k in range(10000):

		cumulative = np.zeros(np.shape(cost))

		for i in range(0, np.shape(cost)[0]):

			aux = probabilities[0][:,i]

			aux = aux.reshape(J0.shape)

			for j in range(1, np.shape(cost)[1]):

				aux = np.concatenate((aux, probabilities[j][:,i].reshape(J0.shape)), axis = 1)

			cumulative += aux * J0[i][0]

		cumulative *= gamma

		cumulative += cost

		Jaux = np.amin(cumulative, axis = 1)

		Jaux = Jaux.reshape(J0.shape)

		if np.all(np.isclose(Jaux, J0)):

			return Best_Policy(Jaux, cost, probabilities, gamma)

		J0 = Jaux

	raise Exception ('Value not found in limit iterations')

def Best_Policy(J, cost, probabilities, gamma):

	cumulative = np.zeros(np.shape(cost))

	for i in range(0, np.shape(cost)[0]):

		aux = probabilities[0][:,i]

		aux = aux.reshape(J.shape)

		for j in range(1, np.shape(cost)[1]):

			aux = np.concatenate((aux, probabilities[j][:,i].reshape(J.shape)), axis = 1)

		cumulative += aux * J[i][0]

	cumulative *= gamma

	cumulative += cost

	policy_Best = np.argmin(cumulative, axis = 1)

	policy = np.zeros(np.shape(cost))

	for i in range(len(policy_Best)):
		policy[i][policy_Best[i]] = 1

	#policy += np.ones(np.shape(policy), dtype = 'int64')

	return policy

def Policy_Iteration(cost, probabilities, gamma):

	pi0 = np.zeros(np.shape(cost))

	pi0 += np.ones(np.shape(pi0))/(np.shape(pi0)[1])

	for i in range(10000):

		Ppi = np.zeros(np.shape(probabilities[0]))

		for j in range(len(probabilities)):

			Ppi += probabilities[j] * pi0[:,j]
			Cpi = np.multiply(cost, pi0)

		Cpi = np.sum(Cpi, axis = 1)

		Cpi = np.array([Cpi])

		Cpi = Cpi.reshape((np.shape(Cpi)[1],1))

		J = np.identity(np.shape(Ppi)[0]) - gamma * Ppi
		
		J = np.linalg.inv(J)

		J = np.dot(J, Cpi)

		best_pol = Best_Policy(J, cost, probabilities, gamma)

		if np.all(np.isclose(pi0, best_pol)):
			return best_pol

		pi0 = best_pol


	raise Exception ('Iterations Exceeded')

#Examples from book, and exercise 18
pa1 = np.array([[0.5, 0.5], [0.5, 0.5]])
pb1 = np.array([[0, 1], [0, 1]])
cost1 = np.array([[0.5, 0], [0.5, 0]])

gamma = 0.99 

probabilities1 = [pa1,pb1]

#print(Value_Iteration(cost1, probabilities1, gamma), 'ValueI 1')

#print(Policy_Iteration(cost1, probabilities1, gamma), 'PolicyI 1')

pa2 = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1]])
pb2 = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 1]])
cost2 = np.array([[1, 0.5], [0, 0], [1, 1]])

probabilities2 = [pa2, pb2]

#print(Value_Iteration(cost2, probabilities2, gamma), 'ValueI 2')

#print(Policy_Iteration(cost2, probabilities2, gamma), 'PolicyI 2')

#Exercise 19

pDOWN19 = np.array([[0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0], [0,0,0,1,0,0,0,0], [0,0,0,0,0,0,0,1], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1] ])
pUP19 = np.array([[0,1,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,1] ])
pSTRAIGHT19 = np.array([[0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,1], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1] ])

cost19 = np.array([[17, 17, 7], [41, 50, 7], [41, 41, 21], [50, 41, 7], [17, 50, 50], [50, 50, 7], [50, 17, 50], [0, 0, 0]])/50

probabilities19 = [pDOWN19, pUP19, pSTRAIGHT19]

#print(Value_Iteration(cost19, probabilities19, 0.99), 'ValueI 19')

#print(Policy_Iteration(cost19, probabilities19, 0.99), 'PolicyI 19') Gives sub-optimal result, why?

#Exercise 21

pH21 = np.zeros((20,19))
last_row = np.ones((20,1))

pH21 = np.concatenate((pH21, last_row), axis = 1)

pNH21 = np.array([[0, 1/2, 1/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

for i in range (3, 18, 2):

	aux = np.zeros((1,20))
	aux[0][i] = 1/((i+3)/2)
	aux[0][i+1] = ((i+3)/2-1)/((i+3)/2)

	pNH21 = np.concatenate((pNH21, aux), axis = 0)
	pNH21 = np.concatenate((pNH21, aux), axis = 0)

aux = np.zeros((1,20))
aux[0][-1] = 1

pNH21 = np.concatenate((pNH21, aux), axis = 0)
pNH21 = np.concatenate((pNH21, aux), axis = 0)
pNH21 = np.concatenate((pNH21, aux), axis = 0)

cost21 = np.zeros((20,1))
cost21[0][0] = 0.9

for i in range(18,0, -2):
	cost21[i-1][0] = (10-(i/2+1))/10
	cost21[i][0] = 1

aux = np.zeros((20,1))
aux[19][0] = 1

cost21 = np.concatenate((cost21, aux), axis = 1)
cost21[19][0] = 0
cost21[19][1] = 0
cost21[18][1] = 1

probabilities21 = [pH21, pNH21]

#print(Value_Iteration(cost21, probabilities21, 0.99), 'ValueI 21')

#print(Policy_Iteration(cost21, probabilities21, 0.99), 'PolicyI 21') 

#Homework

pLEFT = np.array([[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
pRIGHT = np.array([[0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
pUP = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
pDOWN = np.array([[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])

costHW = np.array([[0, 1, 1, 1], [0, 1, 1, 1], [1, 0, 1, 1], [1, 0, 1, 1], [1,1,0,1], [0,0,0,0], [0,0,0,0]])

probabilitiesHW = [pLEFT, pRIGHT, pUP, pDOWN]

#print(Value_Iteration(costHW, probabilitiesHW, 0.9), 'ValueI HW')

#print(Policy_Iteration(costHW, probabilitiesHW, 0.9), 'PolicyI HW')





