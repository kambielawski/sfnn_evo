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
        super().__init__()

        self.Gru = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
        
    def forward(self, pre_neurons, post_neurons, reward, hidden_state, lr):
        Gru_input = torch.cat([pre_neurons, post_neurons, reward])
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
    """
    def __init__(self, neuron_size, input_layer_size, hidden_layer_size, output_layer_size, lr):
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

        self.init_sparsity_matrix()
        self.init_hidden_state_matrix()
        self.init_post_neuron_state_matrix()
        self.lr = nn.Parameter(torch.tensor(lr))

    def set_parameters(self, parameters):
        self.load_state_dict(parameters)

    def get_parameters(self):
        return self.state_dict()

    def init_sparsity_matrix(self):
        """
        initialize nxn adjacency matrix
        This implementation assumes the matrix is ordered as follows [input, hidden, output]
        """
        self.Adjacency_matrix = torch.bernoulli(torch.ones(self.total_neurons, self.total_neurons)/2)
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

    def init_hidden_state_matrix(self):
        self.hidden_state = torch.randn(self.total_neurons, self.total_neurons, self.neuron_size)

    def init_post_neuron_state_matrix(self):
        self.post_neuron_state = torch.randn(self.total_neurons, self.neuron_size)
    
    def forward(self, obs, reward):

        #output placeholder
        placeholder = torch.zeros(self.total_neurons, self.total_neurons, self.neuron_size)
        
        #create observation tensor
        input_signal = obs.reshape(self.input_layer_size, 1).expand(self.input_layer_size, self.neuron_size)
        
        #process input FC layer
        self.post_neuron_state[:self.input_layer_size] = self.input_layer_neuron(input_signal) # [input_layer_size, neuron_size]

        #process input synapse
        for i in range(self.input_layer_size):
            connected_hidden_neurons = (self.Adjacency_matrix[i] == 1).nonzero(as_tuple=True)[0].tolist()
            for j in connected_hidden_neurons:
                placeholder[i,j], self.hidden_state[i,j] = self.input_layer_synapse(self.post_neuron_state[i],
                                                                          self.post_neuron_state[j],
                                                                          reward,
                                                                          self.hidden_state[i,j],
                                                                          self.lr)

        #process hidden FC
        self.post_neuron_state[self.input_layer_size:self.input_layer_size+self.hidden_layer_size] = self.hidden_layer_neuron(placeholder[:, self.input_layer_size:self.input_layer_size+self.hidden_layer_size].sum(axis=0))

        #process hidden synapse
        for i in range(self.hidden_layer_size):
            relative_i = self.input_layer_size + i # relative position with offset
            connected_neurons = (self.Adjacency_matrix[relative_i] == 1).nonzero(as_tuple=True)[0].tolist()
            for j in connected_neurons:
                placeholder[relative_i, j], self.hidden_state[relative_i,j] = self.hidden_layer_synapse(self.post_neuron_state[relative_i],
                                                                      self.post_neuron_state[j],
                                                                      reward,
                                                                      self.hidden_state[relative_i,j], 
                                                                      self.lr)

        #process output FC
        self.post_neuron_state[self.input_layer_size+self.hidden_layer_size:] = self.output_layer_neuron(placeholder[:, self.input_layer_size+self.hidden_layer_size:].sum(axis=0))

        #process ouput (softmax over the first element of each out neuron) 
        for i in range(self.output_layer_size):
            relative_i = i + self.input_layer_size + self.hidden_layer_size
            connected_neurons = (self.Adjacency_matrix[relative_i] == 1).nonzero(as_tuple=True)[0].tolist()
            for j in connected_neurons:
                placeholder[relative_i, j], self.hidden_state[relative_i,j] = self.output_layer_synapse(self.post_neuron_state[relative_i],
                                                                      self.post_neuron_state[j],
                                                                      reward,
                                                                      self.hidden_state[relative_i,j], 
                                                                      self.lr)
                
        return self.post_neuron_state[-self.output_layer_size:, 0].argmax()

if __name__=='__main__':
    model = SFNN(input_layer_size=2, hidden_layer_size=4, output_layer_size=2, neuron_size=4, lr=0.1)
    obs = torch.tensor([5,0.5])
    reward = torch.tensor([5])
    print(model(obs, reward))