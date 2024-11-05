"""
Implementation of a Structurally Flexible Neural Network (SFNN)
"""
import torch
import torch.nn as nn

class Neuron(nn.Module):
    """
    Neuron in SFNN which a fully connected layer. This implementation does not 
    differentiate between the type of neuron (input, hidden, output) 
    
    params:
        - neuron_size: size of the linear layer representing the neuron 

    """
    def __init__(self, neuron_size=4):
        super().__init__()

        self.FC = nn.Linear(neuron_size, neuron_size)
        self.activation = nn.Tanh()

    def forward(self, x):
        return self.activation(self.FC(x))
    
    
class Synapse(nn.Module):
    """
    GRU Synapse to predict the SFNN weights
    note that there is a learnable learning rate to adjust how much the hidden states are updated

    params:
        - input_size: GRU input, typically [pre neuron, post neuron, reward]
        - hidden_size: hidden state of GRU (output of GRU)
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.Gru = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
        
    def forward(self, pre_neurons, post_neurons, reward, hidden_state, lr):
        Gru_input = torch.cat([pre_neurons, post_neurons, reward], axis=1)
        Gru_output = self.Gru(Gru_input, hidden_state)
        updated_hidden_state = hidden_state + Gru_output*lr
        Synapse_output = updated_hidden_state*pre_neurons
        return Synapse_output, updated_hidden_state
    
class SFNN(nn.Module):
    """
    Structurally Flexible Neural Network 

    params:
        - neuron_size: size of FC layer in SFNN neuron
        - input_layer_size: # of neurons in input layer
        - hidden_layer_size: # of neurons in hidden layer
        - ouput_layer_size: # of neurons in output layer
        - lr: learning rate :) this parameter should be optimized by the evo algorithm
        - ticks: number of internal run per forward pass (to process data in the resevoir layers)
    """
    def __init__(self, neuron_size, input_layer_size, hidden_layer_size, output_layer_size, lr, ticks):
        super().__init__()

        self.input_layer_neuron = Neuron(neuron_size=neuron_size)
        self.hidden_layer_neuron = Neuron(neuron_size=neuron_size)
        self.output_layer_neuron = Neuron(neuron_size=neuron_size)
        
        self.input_layer_synapse = Synapse(input_size=neuron_size*2+1, hidden_size=neuron_size)
        self.hidden_layer_synapse = Synapse(input_size=neuron_size*2+1, hidden_size=neuron_size)
        self.output_layer_synapse = Synapse(input_size=neuron_size*2+1, hidden_size=neuron_size)

        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.total_neurons = input_layer_size + hidden_layer_size + output_layer_size
        
        self.neuron_size = neuron_size
        self.ticks = ticks

        self.init_sparsity_matrix()
        self.init_hidden_state_matrix()
        self.init_post_neuron_state_matrix()
        self.lr = nn.Parameter(torch.tensor(lr))

    def set_parameters(self, parameters):
        """
        Loads parameters into the SFNN, should be taken from evo algorithm
        """
        self.load_state_dict(parameters)

    def get_parameters(self):
        """
        returns current parameters 
        """
        return self.state_dict()

    def init_sparsity_matrix(self):
        """
        initialize nxn adjacency matrix
        Assumes the matrix is ordered as follows [input, hidden, output]
        """
        self.Adjacency_matrix = torch.bernoulli(torch.ones(self.total_neurons, self.total_neurons)/2)
        self.vallidate_Adjacency_matrix()

    def vallidate_Adjacency_matrix(self):
        """
        Assert connection as follows:
                 ___
                ↑   ↓
        input → hidden → output
                  ↑________↓
  
        """
        # input cant connect to input
        # input cant connect to output
        self.Adjacency_matrix[:self.input_layer_size, :self.input_layer_size] = 0
        self.Adjacency_matrix[:self.input_layer_size, -self.output_layer_size:] = 0

        # hidden cant connect to input
        self.Adjacency_matrix[self.input_layer_size:self.input_layer_size+self.hidden_layer_size, :self.input_layer_size] = 0

        # output can't connect to input
        # output can't connnect to output
        self.Adjacency_matrix[-self.output_layer_size:, :self.input_layer_size] = 0
        self.Adjacency_matrix[-self.output_layer_size:, -self.output_layer_size:] = 0

        # assert at least one connection from the input layer to hidden
        if not self.Adjacency_matrix[:self.input_layer_size].sum():
            self.Adjacency_matrix[0, self.input_layer_size] = 1

    def init_hidden_state_matrix(self):
        """
        GRU hidden states placeholder (NxNxD)
        """
        self.hidden_state = torch.randn(self.total_neurons, self.total_neurons, self.neuron_size)

    def init_post_neuron_state_matrix(self):
        """
        Neuron values placeholder, after FC layer (NxD)
        """
        self.post_neuron_state = torch.randn(self.total_neurons, self.neuron_size)
    
    def forward(self, obs: torch.tensor, reward: torch.tensor):
        """
         - obs: tensor of shape (obs_dim)
         - reward: tensor of shape (1)
        """

        #synapse output placeholder
        placeholder = torch.zeros(self.total_neurons, self.total_neurons, self.neuron_size) # NxNxD
        
        #create observation tensor
        input_signal = obs.reshape(self.input_layer_size, 1).expand(self.input_layer_size, self.neuron_size) # NxD
        
        for tick in range(self.ticks):
            #process input FC layer
            self.post_neuron_state[:self.input_layer_size] = self.input_layer_neuron(input_signal) # input_layer_size x D

            #process input synapse        
            connected_hidden_neurons = self.Adjacency_matrix[:self.input_layer_size] == 1
            #only reason this loop exists is because torch's GRUCell does not support batched operations!! :@
            for i in range(self.input_layer_size): 
                placeholder[i, connected_hidden_neurons[i]], \
                    self.hidden_state[i, connected_hidden_neurons[i]] = \
                        self.input_layer_synapse(self.post_neuron_state[i].repeat(connected_hidden_neurons[i].sum(), 1), #same pre-neuron across all edges (number_of_connections x D)
                                                self.post_neuron_state[connected_hidden_neurons[i]], #post-neurons (number_of_connections x D)
                                                reward.repeat(connected_hidden_neurons[i].sum(), 1), #rewards (number_of_connections x 1)
                                                self.hidden_state[i, connected_hidden_neurons[i]],   #hidden state (number_of_connections x D)
                                                self.lr)
            
            #process hidden FC (sum all incoming edges then pass to FC)
            self.post_neuron_state[self.input_layer_size : self.input_layer_size + self.hidden_layer_size] = \
                self.hidden_layer_neuron(placeholder[:, self.input_layer_size : self.input_layer_size + self.hidden_layer_size].sum(axis=0))

            #process hidden synapse        
            connected_hidden_neurons = self.Adjacency_matrix[self.input_layer_size : self.input_layer_size + self.hidden_layer_size] == 1
            for i in range(self.hidden_layer_size):
                relative_i = self.input_layer_size + i
                placeholder[relative_i, connected_hidden_neurons[i]], \
                    self.hidden_state[relative_i, connected_hidden_neurons[i]] = \
                        self.hidden_layer_synapse(self.post_neuron_state[relative_i].repeat(connected_hidden_neurons[i].sum(), 1), # (number_of_connections x D)
                                                self.post_neuron_state[connected_hidden_neurons[i]], # (number_of_connections x D)
                                                reward.repeat(connected_hidden_neurons[i].sum(), 1), # (number_of_connection X 1)
                                                self.hidden_state[relative_i, connected_hidden_neurons[i]], # (number_of_connection X D)
                                                self.lr)
            
            #process output FC (sum all incoming edges then pass to FC)
            self.post_neuron_state[-self.output_layer_size:] = \
                self.output_layer_neuron(placeholder[:, -self.output_layer_size:].sum(axis=0))

            #process output synapse
            connected_hidden_neurons = self.Adjacency_matrix[-self.output_layer_size:] == 1
            for i in range(self.output_layer_size):
                relative_i = i + self.input_layer_size + self.hidden_layer_size
                placeholder[relative_i, connected_hidden_neurons[i]], \
                    self.hidden_state[relative_i, connected_hidden_neurons[i]] = \
                        self.output_layer_synapse(self.post_neuron_state[relative_i].repeat(connected_hidden_neurons[i].sum(), 1), # (number_of_connections x D)
                                                  self.post_neuron_state[connected_hidden_neurons[i]], # (number_of_connections x D)
                                                  reward.repeat(connected_hidden_neurons[i].sum(), 1), # (number_of_connections x 1)
                                                  self.hidden_state[relative_i, connected_hidden_neurons[i]], # (number_of_connections x D)
                                                  self.lr) 
                
        #Get last element of each action neuron, then argmax to choose the action
        return self.post_neuron_state[-self.output_layer_size:, 0].argmax()

if __name__=='__main__':
    model = SFNN(input_layer_size=2, hidden_layer_size=4, output_layer_size=2, neuron_size=4, lr=0.1, ticks=1)
    obs = torch.tensor([5,0.5])
    reward = torch.tensor([5])
    # sanity check 
    print(model(obs, reward))