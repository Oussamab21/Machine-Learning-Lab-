import numpy as np
from hmmlearn import hmm

# Use Model in classes

# Names of the states
states = ["Cloudy", "Rainy", "Sunny"]

# States
n_states = len(states)

# Observations
observations = ["Museum", "Beach"]
n_observations = len(observations)

# HMM Model
# with the following parameters
# n_components = number of states
# init_params
# n_iter = number of iterations
model = hmm.MultinomialHMM(n_components=n_states,
                           init_params="",
                           n_iter=10,
                           algorithm='viterbi', tol=0.00001)

model.startprob_ = np.array([1, 0, 0])
model.transmat_ = np.array([
                            [0.2, 0.6, 0.2],
                            [0.3, 0.2, 0.5],
                            [0.1, 0.1, 0.8]
                            ])
model.emissionprob_ = np.array([
                                [0.7, 0.3],
                                [1, 0],
                                [0.1, 0.9]
                                ])
# Generates random samples. 3 in this case
gen1 = model.sample(3)
print(gen1)

# Generates 5 samples: Returns a tuble of sequences generated, and their respective states
seqgen2, stat2 = model.sample(5)
print(seqgen2)
print(stat2)

# Generates 2 samples
gen3 = model.sample(2)
print(gen3)

# Given a Sequence computes the log probability under the current model
# Using the example in class: Probability of going to the Museum and then the Beach
sequence1 = np.array([[0, 1]]).T
logproba = model.score(sequence1)
print(logproba)

# Computes the log probability and compute the posteriors
logproba_extend = model.score_samples(sequence1)
print(logproba_extend)

# Find the most probable states that corresponds to the sequence observed
p = model.predict(sequence1)
print(p)
p_extend = model.score_samples(sequence1)
print(p_extend)

# Trying with multiple sequences
sequence3 = np.array([[0, 1, 0, 1]]).T
sequence4 = np.array([[1, 1, 0, 1, 1]]).T
sample = np.concatenate([sequence3, sequence4])
lengths = [len(sequence3), len(sequence4)]
# Fits the model parameters according to the observed sequence
model.fit(sample, lengths)

# Print neatly the names of the states and use the Viterbi algorithm to find the most likely
# hidden states using the Viterbi algorithm
sequence = np.array([[1, 1, 0, 1]]).T
logprob, state_seq = model.decode(sequence, algorithm="viterbi")
print("Observations", ", ".join(map(lambda x: observations[int(x)], sequence)))
print("Associated states:", ", ".join(map(lambda x: states[x], state_seq)))



