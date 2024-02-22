Hidden Markov Model Implementation
This script provides an implementation of a Hidden Markov Model (HMM), a statistical model that represents systems with hidden states. The model can be used to predict a sequence of hidden states based on a sequence of observed events. This implementation includes methods for running the forward algorithm and the Viterbi algorithm on a given sequence of observation states.

Features
Initialization: Set up an HMM with specified observation states, hidden states, prior probabilities, transition probabilities, and emission probabilities.
Forward Algorithm: Calculate the likelihood of a given sequence of observation states.
Viterbi Algorithm: Determine the most likely sequence of hidden states given a sequence of observation states.

Methods
Initialization
The HiddenMarkovModel class is initialized with the following parameters:

observation_states: An array of observed states in the system.
hidden_states: An array of hidden states in the model.
prior_p: Prior probabilities of each hidden state.
transition_p: Transition probability matrix, representing the probabilities of transitioning from one hidden state to another.
emission_p: Emission probability matrix, representing the probabilities of an observed state being generated from a hidden state.

Forward Algorithm
forward(self, input_observation_states: np.ndarray) -> float

Calculates the forward probability (likelihood) of the observed sequence.

input_observation_states: The sequence of observation states to calculate the likelihood for.
Returns the likelihood of the observed sequence.

Viterbi Algorithm
viterbi(self, decode_observation_states: np.ndarray) -> list

Determines the most likely sequence of hidden states that could have generated the given sequence of observation states.

decode_observation_states: The sequence of observation states to decode.
Returns a list of the most likely sequence of hidden states.


**Usage**
To use this script, instantiate the HiddenMarkovModel class with your model parameters, then call the forward or viterbi methods with your sequence of observation states.

# Example Initialization
hmm = HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, emission_p)

# Run forward algorithm
likelihood = hmm.forward(input_observation_states)

# Run Viterbi algorithm
best_sequence = hmm.viterbi(decode_observation_states)


Requirements
Python 3.x
NumPy
Ensure NumPy is installed in your Python environment to utilize this script.
