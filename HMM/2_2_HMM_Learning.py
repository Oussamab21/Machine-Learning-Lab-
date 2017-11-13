"""
Compute the Probability of abbaa
Apply BaumWelch 1 iteration and check probability of the string
Repeat 15 iterations
Repeat at convergence (150 iterations)
Create a HMM with 5 states with parameters initialized at any non zero correct values. Learn using abbaa
"""

import numpy
from hmmlearn import hmm

# Create Model with the following parameters:
# a = 0
# b = 1
states = 3
model = hmm.MultinomialHMM(n_components=states, init_params="",n_iter=100, algorithm='viterbi', tol=0.00001)
model.startprob_ = numpy.array([0.5, 0.3, 0.2])
model.transmat_ = numpy.array([[0.45, 0.35, 0.2], [0.10, 0.50, 0.4], [0.15, 0.25, 0.6]])
model.emissionprob_ = numpy.array([[1.0, 0], [0.5, 0.5], [0, 1.0]])
sequence = numpy.array([[0, 1, 1, 0, 0]]).T
print(sequence)
Z = model.score(sequence)
print(Z)
# 1. Probability = 0.000026

# Apply Baum Welch 1 iteration
model = hmm.MultinomialHMM(n_components=states, init_params="",n_iter=1, algorithm='viterbi', tol=0.00001)
model.startprob_ = numpy.array([0.5, 0.3, 0.2])
model.transmat_ = numpy.array([[0.45, 0.35, 0.2], [0.10, 0.50, 0.4], [0.15, 0.25, 0.6]])
model.emissionprob_ = numpy.array([[1.0, 0], [0.5, 0.5], [0, 1.0]])
model.fit(sequence)
Z = model.score(sequence)
print(Z)
# 2. Probability = 0.001151

# Apply Baum Welch 15 iteration
model = hmm.MultinomialHMM(n_components=states, init_params="",n_iter=15, algorithm='viterbi', tol=0.00001)
model.startprob_ = numpy.array([0.5, 0.3, 0.2])
model.transmat_ = numpy.array([[0.45, 0.35, 0.2], [0.10, 0.50, 0.4], [0.15, 0.25, 0.6]])
model.emissionprob_ = numpy.array([[1.0, 0], [0.5, 0.5], [0, 1.0]])
model.fit(sequence)
Z = model.score(sequence)
print(Z)
# 3. Probability = 0.04

# Apply Baum Welch at convergence
model = hmm.MultinomialHMM(n_components=states, init_params="",n_iter=200, algorithm='viterbi', tol=0.00001)
model.startprob_ = numpy.array([0.5, 0.3, 0.2])
model.transmat_ = numpy.array([[0.45, 0.35, 0.2], [0.10, 0.50, 0.4], [0.15, 0.25, 0.6]])
model.emissionprob_ = numpy.array([[1.0, 0], [0.5, 0.5], [0, 1.0]])
model.fit(sequence)
Z = model.score(sequence)
print(Z)
# 4. Probability = 0.041

# Learn with 5 states
states = 5
sequence = numpy.array([[0, 1, 1, 0, 0]]).T
model = hmm.MultinomialHMM(n_components=states, init_params="",n_iter=200, algorithm='viterbi', tol=0.00001)
model.startprob_ = numpy.array([0.3, 0.3, 0.2, 0.1, 0.1])
model.transmat_ = numpy.array([[0.4, 0.2, 0.2, 0.1, 0.1], [0.4, 0.2, 0.2, 0.1, 0.1], [0.4, 0.2, 0.2, 0.1, 0.1], [0.4, 0.2, 0.2, 0.1, 0.1], [0.4, 0.2, 0.2, 0.1, 0.1]])
model.emissionprob_ = numpy.array([[1.0, 0], [0.5, 0.5], [0, 1.0], [0.5, 0.5], [0.5, 0.5]])
model.fit(sequence)
Z = model.score(sequence)
print("After 5 states")
print(Z)
# 5. Probability = 1 at convergence






