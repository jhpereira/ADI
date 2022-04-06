#!/usr/bin/env python
# coding: utf-8

# # Learning and Decision Making

# ## Laboratory 3: Partially observable Markov decision problems
# 
# In the end of the lab, you should export the notebook to a Python script (File >> Download as >> Python (.py)). Your file should be named `padi-lab3-groupXX.py`, where the `XX` corresponds to your group number and should be submitted to the e-mail <adi.tecnico@gmail.com>. 
# 
# Make sure...
# 
# * **... that the subject is of the form `[<group n.>] LAB <lab n.>`.** 
# 
# * **... to strictly respect the specifications in each activity, in terms of the intended inputs, outputs and naming conventions.** 
# 
# In particular, after completing the activities you should be able to replicate the examples provided (although this, in itself, is no guarantee that the activities are correctly completed).
# 
# ### 1. The POMDP model
# 
# Consider once again the Pacman modeling problem described in the Homework and for which you wrote a Markov decision problem model. In this lab, you will consider a larger version of the Pacman problem, described by the diagram:
# 
# <img src="pacman-big.png">
# 
# Recall that the POMDP should describe the decision-making of a player. In the above domain,
# 
# * The ghost **moves randomly between cells 1-3**.
# * The player controls the movement of Pacman through four actions: `Up`, `Down`, `Left`, and `Right`. 
# * Each action moves the Pacman character one step in the corresponding direction, if an adjacent cell exists in that direction. Otherwise, Pacman remains in the same place.
# * The cell in the bottom left corner (cell `29`) is adjacent, to the left, to the cell in the bottom right corner (cell `35`). In other words, if Pacman "moves left" in cell `29` it will end up in cell `35` and vice-versa.
# * If Pacman lies in the same cell as the ghost (in either cell `1`, `2`, or `3`), the player loses the game. However, if Pacman "eats" the blue pellet (in cell `24`), it gains the ability to "eat" the ghost. In this case, if Pacman lies in the same cell as the ghost, it "eats" the ghost and wins the game. Assume that Pacman can never be in cell `24` without "eating" the pellet.
# * Pacman is unable to see the ghost unless if it stands in the same position as the ghost (however, it does know its own position and whether it ate the pellet or not).
# 
# In this lab you will use a POMDP based on the aforementioned domain and investigate how to simulate a partially observable Markov decision problem and track its state. You will also compare different MDP heuristics with the optimal POMDP solution.
# 
# **Throughout the lab, unless if stated otherwise, use $\gamma=0.9$.**
# 
# $$\diamond$$
# 
# In this first activity, you will implement an POMDP model in Python. You will start by loading the POMDP information from a `numpy` binary file, using the `numpy` function `load`. The file contains the list of states, actions, observations, transition probability matrices, observation probability matrices, and cost function.

# ---
# 
# #### Activity 1.        
# 
# Write a function named `load_pomdp` that receives, as input, a string corresponding to the name of the file with the POMDP information, and a real number $\gamma$ between $0$ and $1$. The loaded file contains 6 arrays:
# 
# * An array `X` that contains all the states in the POMDP, represented as strings. In the Pacman environment above, for example, there is a total of 209 states, each describing the position of Pacman in the environment, whether it has eaten the blue pellet, and the position of the ghost. Those states are either one of the strings `"V"` or `"D"`, corresponding to the absorbing "victory" and "defeat" states, or a string of the form `"(p, s, g)"`, where:
#     * `p` is a number between 1 and 35 indicating the position of Pacman;
#     * `s` is either `0` or `S`, where `0` indicates that Pacman has not yet eaten the pellet; `S` indicates that Pacman has eaten the pellet (and now has "superpowers");
#     * `g` is a number between 1 and 3, indicating the position of the ghost.
# * An array `A` that contains all the actions in the MDP, also represented as strings. In the Pacman environment above, for example, each action is represented as a string `"Up"`, `"Down"`, `"Left"` or `"Right"`.
# * An array `Z` that contains all the observations in the POMDP, also represented as strings. In the Pacman environment above, for example, there is a total of 77 observations, each describing the position of Pacman in the environment, whether it has eaten the blue pellet, and whether it sees the ghost. It also observes the victory and defeat states. This means that the strings are either `"V"` or `"D"`, corresponding to the "victory" and "defeat" states, or a string of the form `"(p, s, g)"`, where:
#     * `p` is a number between 1 and 35 indicating the position of Pacman;
#     * `s` is either `0` or `S`, where `0` indicates that Pacman has not yet eaten the pellet; `S` indicates that Pacman has eaten the pellet (and now has "superpowers");
#     * `g` is a number between 0 and 3, 0 indicating that the ghost is not seen, and the numbers between 1 and 3 indicates the position of the ghost (when visible).
# * An array `P` containing `len(A)` subarrays, each with dimension `len(X)` &times; `len(X)` and  corresponding to the transition probability matrix for one action.
# * An array `O` containing `len(A)` subarrays, each with dimension `len(X)` &times; `len(Z)` and  corresponding to the observation probability matrix for one action.
# * An array `c` containing the cost function for the POMDP.
# 
# Your function should create the POMDP as a tuple `(X, A, Z, (Pa, a = 0, ..., len(A)), (Oa, a = 0, ..., len(A)), c, g)`, where `X` is a tuple containing the states in the POMDP represented as strings (see above), `A` is a tuple containing the actions in the POMDP represented as strings (see above), `Z` is a tuple containing the observations in the POMDP represented as strings (see above), `P` is a tuple with `len(A)` elements, where `P[a]` is an `np.array` corresponding to the transition probability matrix for action `a`, `O` is a tuple with `len(A)` elements, where `O[a]` is an `np.array` corresponding to the observation probability matrix for action `a`, `c` is an `np.array` corresponding to the cost function for the POMDP, and `g` is a float, corresponding to the discount and provided as the argument $\gamma$ of your function. Your function should return the POMDP tuple.
# 
# ---

# In[7]:


import numpy as np
import numpy.random as rand

def load_pomdp(file_str, gamma):
    
    gamma = np.clip(gamma, 0, 1)
    
    data = np.load(file_str)
    
    M = []
    
    for k in data.keys():
        M.append(data[k])
        
    M.append(gamma)
    
    M[1] = tuple(M[1])
    
    M = tuple(M)
        
    return M


# ### 2. Sampling
# 
# You are now going to sample random trajectories of your POMDP and observe the impact it has on the corresponding belief.

# ---
# 
# #### Activity 2.
# 
# Write a function called `gen_trajectory` that generates a random POMDP trajectory using a uniformly random policy. Your function should receive, as input, a POMDP described as a tuple like that from **Activity 1** and two integers, `x0` and `n` and return a tuple with 3 elements, where:
# 
# 1. The first element is a `numpy` array corresponding to a sequence of `n+1` state indices, $x_0,x_1,\ldots,x_n$, visited by the agent when following a uniform policy (i.e., a policy where actions are selected uniformly at random) from state with index `x0`. In other words, you should select $x_1$ from $x_0$ using a random action; then $x_2$ from $x_1$, etc.
# 2. The second element is a `numpy` array corresponding to the sequence of `n` action indices, $a_0,\ldots,a_{n-1}$, used in the generation of the trajectory in 1.;
# * The third element is a `numpy` array corresponding to the sequence of `n` observation indices, $z_1,\ldots,z_n$, experienced by the agent during the trajectory in 1.
# 
# The `numpy` array in 1. should have a shape `(n+1,)`; the `numpy` arrays from 2. and 3. should have a shape `(n,)`.
# 
# **Note:** Your function should work for **any** POMDP specified as above.
# 
# ---

# In[8]:


def gen_trajectory(M, x0, n):
    
    action_list = np.arange(len(M[1]), dtype=int) #all actions

    state_list = np.arange(np.shape(M[0])[0], dtype=int)
    
    obs_list = np.arange(np.shape(M[2])[0], dtype=int)

    old_state = state_list[x0]
    
    trajectory = np.zeros(n+1, dtype=int)
    
    actions = np.zeros(n, dtype=int)
    
    observations = np.zeros(n, dtype=int)
    
    for i in range(n):
        
        action = np.random.choice(len(action_list))
        
        trajectory[i] = old_state

        actions[i] = action

        new_state = np.random.choice(state_list, p = M[3][action][old_state,:])
        
        obs_prob = M[4][action][new_state,:]
        
        observation = np.random.choice(obs_list, p = obs_prob)

        observations[i] = observation
        
        old_state = new_state
        
    trajectory[i+1] = new_state
                
    info = tuple((trajectory, actions, observations))
          
    return info


# You will now write a function that samples a given number of possible belief points for a POMDP. To do that, you will use the function from **Activity 2**.
# 
# ---
# 
# #### Activity 3.
# 
# Write a function called `sample_beliefs` that receives, as input, a POMDP described as a tuple like that from **Activity 1** and an integer `n`, and return a tuple with `n+1` elements **or less**, each corresponding to a possible belief state (represented as a $1\times|\mathcal{X}|$ vector). To do so, your function should
# 
# * Generate a trajectory with `n` steps from a random initial state, using the function `gen_trajectory` from **Activity 2**.
# * For the generated trajectory, compute the corresponding sequence of beliefs, assuming that the agent does not know its initial state (i.e., the initial belief is the uniform belief, and should also be considered). 
# 
# Your function should return a tuple with the resulting beliefs, **ignoring duplicate beliefs or beliefs whose distance is smaller than $10^{-3}$.**
# 
# **Suggestion:** You may want to define an auxiliary function `belief_update` that receives a POMDP, a belief, an action and an observation and returns the updated belief.
# 
# **Note:** Your function should work for **any** POMDP specified as above. To compute the distance between vectors, you may find useful `numpy`'s function `linalg.norm`.
# 
# 
# ---

# In[9]:


def belief_update(b, P, O):
    
    b = np.matmul(b,P)*O.T
    
    return b/np.linalg.norm(b,1) 

def sample_beliefs(M, n):
    
    P = M[3]
    O = M[4]
        
    X = M[0]
    
    s0 = rand.choice(len(X))
    
    t = gen_trajectory(M, s0, n)
    
    b0 = np.ones(len(X))/len(X)
    
    B = np.zeros([n+1, *b0.shape])
    
    B[0] = b0
    
    B_unique = []
    
    B_unique.append(b0)
    
    for i in range(0, n):
        a = t[1][i]
        z = t[2][i]
        B[i+1] = belief_update(B[i], P[a], O[a][:,z])
        
        not_duplicate = True
        
        for j in range(0, i):
            if np.linalg.norm(B[j]-B[i+1], 2) < 1e-3:
                not_duplicate = False
                
        if not_duplicate:
            B_unique.append(B[i+1])
            
    B_unique = np.asarray(B_unique)
    
    B_unique = np.reshape(B_unique, [B_unique.shape[0],1,B_unique.shape[1]])
    
    return B_unique


# ### 3. Solution methods
# 
# In this section you are going to compare different solution methods for POMDPs discussed in class.

# ---
# 
# #### Activity 4
# 
# Write a function `solve_mdp` that takes as input a POMDP represented as a tuple like that of **Activity 1** and returns a `numpy` array corresponding to the **optimal $Q$-function for the underlying MDP**. Stop the algorithm when the error between iterations is smaller than $10^{-8}$.
# 
# **Note:** Your function should work for **any** POMDP specified as above. You may reuse code from previous labs.
# 
# ---

# In[10]:


def solve_mdp(M):
    
    c = M[-2]
    P = M[3]
    gamma = M[-1]
    
    c = c.T.reshape([c.shape[1], c.shape[0], 1])
    
    pi = np.ones(c.shape)/c.shape[0]
    
    k = 0
    
    Q_prev = np.ones([*c.shape])
    Q = np.zeros([*c.shape])
    
    while not np.allclose(Q,Q_prev):
        
        Q_prev = Q
        
        P_pi = np.sum(P*pi, axis = 0)
        c_pi = np.sum(c*pi, axis = 0)
        
        J = np.matmul(np.linalg.inv(np.eye(P_pi.shape[0]) - gamma*P_pi), c_pi)
        
        Q = c + gamma * np.matmul(P,J)
        
        pi = np.isclose(Q, np.min(Q,axis = 0)).astype(int)

        pi = pi/np.sum(pi, axis = 0)
        
        k += 1
    
    Q = Q.T.reshape([Q.shape[1],Q.shape[0]])
    
    return Q


# ---
# 
# #### Activity 5
# 
# You will now test the different MDP heuristics discussed in class. To that purpose, write down a function that, given a belief vector and the solution for the underlying MDP, computes the action prescribed by each of the three MDP heuristics. In particular, you should write down a function named `get_heuristic_action` that receives, as inputs:
# 
# * A belief state represented as a `numpy` array like those of **Activity 3**;
# * The optimal $Q$-function for an MDP (computed, for example, using the function `solve_mdp` from **Activity 4**);
# * A string that can be either `"mls"`, `"av"`, or `"q-mdp"`;
# 
# Your function should return an integer corresponding to the index of the action prescribed by the heuristic indicated by the corresponding string, i.e., the most likely state heuristic for `"mls"`, the action voting heuristic for `"av"`, and the $Q$-MDP heuristic for `"q-mdp"`. *In all heuristics, ties should be broken randomly, i.e., when maximizing/minimizing, you should randomly select between all maximizers/minimizers*.
# 
# ---

# In[11]:


def get_heuristic_action(belief, Q, heur_str):

        Q = Q.T.reshape([Q.shape[1], Q.shape[0], 1])
        
        pi = np.isclose(Q, np.min(Q,axis = 0)).astype(int)

        pi = pi/np.sum(pi, axis = 0)

        if heur_str == 'mls':

            pi = np.concatenate(pi, axis = 1)
            
            MLS = np.isclose(b, np.max(b)).astype(int) #Actually random
            
            MLS = MLS/np.sum(MLS)            

            MLS = np.random.choice(MLS.shape[1], p = MLS[0])

            pi_MLS = pi[MLS,:]
                        
            pi_MLS = np.isclose(pi_MLS, np.max(pi_MLS)).astype(int)
            
            pi_MLS = pi_MLS/np.sum(pi_MLS)
                                                
            return np.random.choice(np.arange(len(pi_MLS)), p = pi_MLS)
        
        if heur_str == "av":

            pi = np.concatenate(pi, axis = 1)
            
            pi = belief@pi

            pi = np.isclose(pi, np.max(pi)).astype(int)
                        
            pi = pi/np.sum(pi)
            
            return (np.random.choice(len(pi[0]), p = pi[0]))
        
        if heur_str == "q-mdp":

            pi = belief@Q
            
            pi = np.isclose(pi, np.min(pi)).astype(int)
            
            pi = pi/np.sum(pi)
            
            action_list = np.arange(np.shape(pi)[0])
                                
            return (np.random.choice(action_list, p = pi[:,0,0])).astype(int)


# Suppose that the optimal cost-to-go function for the POMDP can be represented using a set of $\alpha$-vectors that have been precomputed for you. 
# 
# ---
# 
# #### Activity 6
# 
# Write a function `get_optimal_action` that, given a belief vector and a set of pre-computed $\alpha$-vectors, computes the corresponding optimal action. Your function should receive, as inputs,
# 
# * A belief state represented as a `numpy` array like those of **Activity 3**;
# * The set of optimal $\alpha$-vectors, represented as a `numpy` array `av`; the $\alpha$-vectors correspond to the **columns** of `av`;
# * A list `ai` containing the **indices** (not the names) of the actions corresponding to each of the $\alpha$-vectors. In other words, the `ai[k]` is the action index of the $\alpha$-vector `av[:, k]`.
# 
# Your function should return an integer corresponding to the index of the optimal action. *Ties should be broken randomly, i.e., when selecting the minimizing action, you should randomly select between all minimizers*.
# 
# ---

# In[12]:


def get_optimal_action(b, av, ai):
    
    W = np.matmul(b, av)
    
    index_list = np.where(W == np.min(W))
    
    new_list = []
    
    for x in index_list:
        new_list.append(x[0])
        
    actions = []
    
    for x in new_list:
        actions.append(ai[x])
        
    action = rand.choice(actions)
    
    return action

