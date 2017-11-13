"""
Create a HMM using the model
Learn the HMM with this samples:
aaabb
abaabbb
aaababb
aabab
ab

Learn the HMM with another sample
bbbaa
babbaa
bbbabaa
bbabba
bbaa

Compute the probabilities of aababbb, bbabaaa
"""


import numpy
from hmmlearn import hmm
# Create Model with the following parameters:
# a = 0
# b = 1
states = 2
model_a = hmm.MultinomialHMM(n_components=states, init_params="",n_iter=100, algorithm='viterbi', tol=0.00001)
model_a.startprob_ = numpy.array([0.31, 0.69])
model_a.transmat_ = numpy.array([[0.45, 0.55], [0.52, 0.48]])
model_a.emissionprob_ = numpy.array([[0.48, 0.51], [0.40, 0.6]])
sequence_a1 = [[0], [0], [0], [1], [1]]
sequence_a2 = [[0], [1], [0], [0], [1], [1], [1]]
sequence_a3 = [[0], [0], [0], [1], [0], [1], [1]]
sequence_a4 = [[0], [0], [1], [0], [1]]
sequence_a5 = [[0], [1]]
sequence_a = numpy.concatenate([sequence_a1, sequence_a2, sequence_a3, sequence_a4, sequence_a5
                                ])
print("A before training: ")
print(model_a.transmat_)
print(model_a.startprob_)
print(model_a.emissionprob_)
model_a.fit(sequence_a, [5, 7, 7, 5, 2])
print("A After training")
print(model_a.transmat_)
print(model_a.startprob_)
print(model_a.emissionprob_)

# Model b
states = 2
model_b = hmm.MultinomialHMM(n_components=states, init_params="",n_iter=100, algorithm='viterbi', tol=0.00001)
model_b.startprob_ = numpy.array([0.31, 0.69])
model_b.transmat_ = numpy.array([[0.45, 0.55], [0.52, 0.48]])
model_b.emissionprob_ = numpy.array([[0.48, 0.51], [0.40, 0.6]])
sequence_b1 = [[1], [1], [1], [0], [0]]
sequence_b2 = [[1], [0], [1], [1], [0], [0], [0]]
sequence_b3 = [[1], [1], [1], [0], [1], [0], [0]]
sequence_b4 = [[1], [1], [0], [1], [1], [0]]
sequence_b5 = [[1], [1], [0], [0]]
sequence_b = numpy.concatenate([sequence_b1, sequence_b2, sequence_b3, sequence_b4, sequence_b5
                                ])
print("B before training: ")
print(model_b.transmat_)
print(model_b.startprob_)
print(model_b.emissionprob_)
model_a.fit(sequence_b, [5, 7, 7, 6, 4])
print("B After training")
print(model_b.transmat_)
print(model_b.startprob_)
print(model_b.emissionprob_)

# Predicts some results:
sequence_1 = numpy.array([[0, 0, 1, 0, 1, 1, 1]]).T
sequence_2 = numpy.array([[1, 1, 0, 1, 0, 0, 0]]).T

Z1 = model_a.score(sequence_1)
Z2 = model_a.score(sequence_2)
print("Scores for Model A: ")
print(Z1, Z2)
# Probabilities are almost 0 and 0.00023

Z3 = model_b.score(sequence_1)
Z4 = model_b.score(sequence_2)
print("Scores for Model B: ")
print(Z3, Z4)
# Probabilities: are 0.000015 and almost zero







