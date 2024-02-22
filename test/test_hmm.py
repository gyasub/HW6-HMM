import pytest
from hmm import HiddenMarkovModel
import numpy as np




def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """
    # Loading the dictionaries
    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    mini_hidden_states = mini_hmm['hidden_states']
    mini_observation_states = mini_hmm['observation_states']
    mini_prior_p = mini_hmm['prior_p']
    mini_transition_p = mini_hmm['transition_p']
    mini_emission_p = mini_hmm['emission_p']

    mini_obs_seq = mini_input['observation_state_sequence']
    mini_best_seq = mini_input['best_hidden_state_sequence']

    # Initializing the Hidden Markov Model
    my_hmm = HiddenMarkovModel(mini_observation_states, mini_hidden_states, mini_prior_p, mini_transition_p, mini_emission_p)


    # Running the Forward Algorithm
    forward_probability = my_hmm.forward(mini_obs_seq)

    # Asserting the value of the forward probability
    assert np.allclose(forward_probability, 0.035, atol= 0.001)
    

    # Running the Viterbi Algorithm
    viterbi_seq = my_hmm.viterbi(mini_obs_seq)


    # Asserting the correct sequence output 
    assert np.all(viterbi_seq == mini_best_seq)
    
    
    # Edge cases
    # Assert that prior probabilities are not zero 
    assert not np.any(mini_prior_p == 0)

    # Asserting that transition probabilities sum to 1 
    rows_sum_to_one = np.allclose(np.sum(mini_transition_p, axis=1), np.ones(mini_transition_p.shape[0]))
    assert rows_sum_to_one

    # Closing the NPZ files
    mini_hmm.close()
    mini_input.close()

def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    # Loading the dictionaries
    full_hmm=np.load('./data/full_weather_hmm.npz')
    full_input=np.load('./data/full_weather_sequences.npz')

    full_hidden_states = full_hmm['hidden_states']
    full_observation_states = full_hmm['observation_states']
    full_prior_p = full_hmm['prior_p']
    full_transition_p = full_hmm['transition_p']
    full_emission_p = full_hmm['emission_p']

    full_obs_seq = full_input['observation_state_sequence']
    full_best_seq = full_input['best_hidden_state_sequence']

    
    # Initializing my Hidden Markov Model

    my_hmm = HiddenMarkovModel(full_observation_states, full_hidden_states, full_prior_p, full_transition_p, full_emission_p)

    # Running the Forward Algorithm
    forward_probability = my_hmm.forward(full_obs_seq)

    # Asserting the value of the forward probability
    assert np.allclose(forward_probability, 1.6864513843961343e-11, atol= 0.001)

    # Running the Viterbi Algorithm
    viterbi_seq = my_hmm.viterbi(full_obs_seq)

    # Asserting the correct sequence output 
    assert np.all(viterbi_seq == full_best_seq)
    assert len(viterbi_seq) == len(full_best_seq)

    # Closing the NPZ files
    full_hmm.close()
    full_input.close()
    









