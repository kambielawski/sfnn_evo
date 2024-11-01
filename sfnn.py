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
    
    def get_params(self):
        return [param for param in self.parameters() if param.requires_grad]
    
class Synapse(nn.Module):
    """
    GRU Synapse to predict the SFNN weights
    note that there is a learnable learning rate to adjust how much the hidden states are updated

    params:
        - input_size: GRU input, typically [pre neuron, post neuron, reward]
        - hidden_size: hidden state of GRU (output of GRU)
    """
    def __init__(self, input_size, hidden_size):
        #TODO: MAY NEED TO MODIFY THIS TO MAKE IT ACCEPT A LAYER HIDDEN STATE
        super().__init__()

        self.Gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.hidden_state = torch.randn(hidden_size) #MAY NEED TO CHANGE BASED ON TYPE OF SYNAPSE!
        self.lr = nn.Parameter(torch.tensor(0.01))

    def forward(self, pre_neurons, post_neurons, reward):
        Gru_input = torch.cat([pre_neurons, post_neurons, reward])
        self.hidden_state = self.hidden_state + self.Gru(Gru_input, self.hidden_state)*self.lr
        post_neurons = self.hidden_state*pre_neurons
        return post_neurons
    
class SFNN(nn.Module):
    """
    Structurally Flexible Neural Network 

    params:
        - neuron_size: size of FC layer in SFNN neuron
        - input_layer_size: # of neurons in input layer
        - hidden_layer_size: # of neurons in hidden layer
        - ouput_layer_size: # of neurons in output layer
    """
    def __init__(self, neuron_size, input_layer_size, hidden_layer_size, output_layer_size):
        super().__init__()

        self.input_layer_neuron = Neuron(neuron_size=neuron_size)
        self.hidden_layer_neuron = Neuron(neuron_size=neuron_size)
        self.outout_layer_neuron = Neuron(neuron_size=neuron_size)
        
        self.input_layer_synapse = Synapse(input_size=neuron_size*2+1, hidden_size=neuron_size)
        self.hidden_layer_synapse = Synapse(input_size=neuron_size*2+1, hidden_size=neuron_size)
        self.output_layer_synapse = Synapse(input_size=neuron_size*2+1, hidden_size=neuron_size)

        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size

        self.init_sparsity_matrix()
        #TODO: add placeholder for synapse states at each edge in Adjacency matrix

    def init_sparsity_matrix(self):
        """
        initialize nxn adjacency matrix
        This implementation assumes the matrix is ordered as follows [input, hidden, output]
        """
        Total_neurons = self.input_layer_size + self.hidden_layer_size + self.output_layer_size
        self.Adjacency_matrix = torch.bernoulli(torch.ones(Total_neurons, Total_neurons)/2)
        self.vallidate_Adjacency_matrix()

    def vallidate_Adjacency_matrix(self):
        # input cant connect to input
        # input cant connect to output
        self.Adjacency_matrix[:self.input_layer_size, :self.input_layer_size] = 0
        self.Adjacency_matrix[:self.input_layer_size, self.input_layer_size+self.hidden_layer_size:] = 0

        # hidden cant connect to input
        self.Adjacency_matrix[self.input_layer_size:self.input_layer_size+self.hidden_layer_size, :self.input_layer_size] = 0

        # output can't connect to input
        # output can't connnect to output
        self.Adjacency_matrix[self.input_layer_size+self.hidden_layer_size:, :self.input_layer_size] = 0
        self.Adjacency_matrix[self.input_layer_size+self.hidden_layer_size:, self.input_layer_size+self.hidden_layer_size:] = 0

    def forward(self, x):
        #TODO: calculate the forward pass according to the adjacency matrix
        #      you will probably need to safe the states of synapses in some dict or matrix!!!
        pass

    def get_parameters(self):
        # TODO: return list of parameters to be optimized in a neat way alligned with evo.py
        pass