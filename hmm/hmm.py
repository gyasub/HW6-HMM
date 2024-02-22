import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        
        # Step 1. Initialize variables
        
        # Input observation states length
        T = len(input_observation_states)
        
        # Hidden States length
        N = len(self.hidden_states)


        # Alpha matrix to record probabilities
        alpha = np.zeros((N,T))

        # Index of the first observation state
        first_obs_index = self.observation_states_dict[input_observation_states[0]]
        
        # Initializing the probabilities at the first observation state in alpha
        alpha[:,0] = self.prior_p * self.emission_p[:, first_obs_index]
       
        # Step 2. Calculate probabilities

        # Recursion to find rest of the probabilities 

        for t in range(1, T):       # Loop over observation sequence
            for j in range(N):      # Loop over hidden states options

                # sum of previous alpha values times their transition probabilities for each hidden state
                alpha_j = np.dot(alpha[:, t-1], self.transition_p[:, j])
                
                # Retrive observation state index for for state we are in 
                obs_index = self.observation_states_dict[input_observation_states[t]]

                # Input probabilities into the matrix by multiplying with the emission probabilities of that state
                alpha[j, t] = alpha_j * self.emission_p[j, obs_index]


        # Step 3. Return final probability 
        forward_prob = np.sum(alpha[:,-1])

        return forward_prob


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        
        # Step 1. Initialize variables
        
        # Input observation states length
        T = len(decode_observation_states)
        
        # Hidden States length
        N = len(self.hidden_states)

        # Store probabilities of hidden state at each step 
        viterbi_table = np.zeros((N, T))

        # Store best path for traceback
        best_path = np.zeros(T, dtype=int)
        path_trace = np.zeros((N,T), dtype=int)

        # Index of the first observation state
        first_obs_index = self.observation_states_dict[decode_observation_states[0]]         
        
        # Assign values to first column of viterbi table
        viterbi_table[:, 0] = self.prior_p * self.emission_p[:,first_obs_index]

        # Step 2. Calculate Probabilities

        for t in range(1, T):       # Loop over observation sequence
            for j in range(N):      # Loop over hidden states options

                # Retrieving index for the current iterations state
                obs_index = self.observation_states_dict[decode_observation_states[t]]
                
                # Calculate the max probability for each state
                probabilities = viterbi_table[:, t-1] * self.transition_p[:, j] * self.emission_p[j, obs_index]
                viterbi_table[j, t] = np.max(probabilities)
                path_trace[j, t] = np.argmax(probabilities)
        

            
        # Step 3. Traceback

        # Assign last item in array to the final highest index
        best_path[-1] = np.argmax(viterbi_table[:, T-1])
        
        
        # Tracing back best path
        for t in range(T-2, -1, -1):
            best_path[t] = path_trace[int(best_path[t+1]), t+1]
        
        
        
        # Step 4. Return best hidden state sequence
        # Convert indices to state names
        best_sequence = [self.hidden_states[i] for i in best_path]
        best_array = np.array(best_sequence)
        
        return best_array