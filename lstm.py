# This is the implemenation of basic LSTM
import numpy as np


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


class LSTM_state:
    ''' Class holding all the states of LSTM unit '''

    def __init__(self, input_dim, num_mem_cells):
        '''At time t=0 initialize the states to zeros'''

        self.g = np.zeros(num_mem_cells)
        self.i = np.zeros(num_mem_cells)
        self.f = np.zeros(num_mem_cells)
        self.o = np.zeros(num_mem_cells)
        self.s = np.zeros(num_mem_cells)
        self.h = np.zeros(num_mem_cells)


class LSTM_data:

    def process_data(self, input_txt_file):
        ''' Process the input data
        @params:
        --------
            input_txt_file - text file input containing the test characters
        '''

        data = []
        with open(input_txt_file, 'r') as f:
            data = f.read()

        chars = list(set(data))  # list of unique characters
        vocab_size = len(chars)

        #  Characters to index and vice versa
        self.char_to_index = {c: i for i, c in enumerate(chars)}
        self.index_to_char = {i: c for i, c in enumerate(chars)}

        self.vocab_size = vocab_size
        self.data = data
        self.data_len = len(data)


class LSTM_param:
    ''' Class holding all the LSTM learnable weights and biases of LSTM unit'''

    def random_array(self, mu, sigma, *shape_args):
        np.random.seed(0)
        return np.random.rand(*shape_args) * sigma + mu

    def init_param(self, num_mem_cells, concat_dim, mu=-0.1, sigma=0.2):
        '''initialize the weights'''

        # Weight initialization
        self.wg = self.random_array(mu, sigma, num_mem_cells, concat_dim)
        self.wi = self.random_array(mu, sigma, num_mem_cells, concat_dim)
        self.wf = self.random_array(mu, sigma, num_mem_cells, concat_dim)
        self.wo = self.random_array(mu, sigma, num_mem_cells, concat_dim)

        # Bias initialization
        self.bg = self.random_array(mu, sigma, num_mem_cells)
        self.bi = self.random_array(mu, sigma, num_mem_cells)
        self.bf = self.random_array(mu, sigma, num_mem_cells)
        self.bo = self.random_array(mu, sigma, num_mem_cells)

        # weight gradient initialization
        self.dwg = np.zeros_like(self.wg)
        self.dwi = np.zeros_like(self.wi)
        self.dwf = np.zeros_like(self.wf)
        self.dwo = np.zeros_like(self.wo)

        # bias gradient initialization
        self.dbg = np.zeros_like(self.bg)
        self.dbi = np.zeros_like(self.bi)
        self.dbf = np.zeros_like(self.bf)
        self.dbo = np.zeros_like(self.bo)

    def __init__(self, input_dim, num_mem_cells=100, lr=0.1):
        ''' Initialize weights, bias,  character indexing
        @params:
        --------
            input_dim - length of the input sequence to the lstm
            num_mem_cells(optional)  - Number of memory cells, default num_mem_cells = 100
            lr(optional) - learning rate, default lr = 0.1
        '''

        self.lr = lr
        self.num_mem_cells = num_mem_cells
        self.input_dim = input_dim
        self.concat_dim = input_dim + num_mem_cells

        # Init parameters
        self.init_param(num_mem_cells, self.concat_dim)


class LSTM_Node:
    ''' class LSTM_nodes holds the lstm state, parameters'''

    def __init__(self, lstm_param, lstm_state):
        # store reference to parameters and to activations
        self.state = lstm_state
        self.param = lstm_param
        # non-recurrent input to node
        self.x = None
        # non-recurrent input concatenated with recurrent input
        self.xc = None

    def forward_pass(self, x, s_prev=None, h_prev=None):
        ''' LSTM forward pass'''

        # At time t = 0
        if s_prev is None:
            s_prev = np.zeros_like(self.state.s)
        if h_prev is None:
            h_prev = np.zeros_like(self.state.h)

        # save the state
        self.s_prev = s_prev
        self.h_prev = h_prev

        # input concantenation
        xc = np.hstack((x, h_prev))
        self.state.g = np.tanh(np.dot(self.param.wg, xc) + self.param.bg)
        self.state.i = sigmoid(np.dot(self.param.wi, xc) + self.param.bi)
        self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)
        self.state.o = sigmoid(np.dot(self.param.wo, xc) + self.param.bo)
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f
        self.state.h = self.state.s * self.state.o

        # store the inputs
        self.x = x
        self.xc = xc


class LSTM_network:

    def __init__(self, input_dim, num_mem_cells=100, learning_rate=0.1):
        '''Initialize the LSTM unit, LSTM state'''

        # weights and bias are reused, so initialize lstm_param only once
        self.lstm_param = LSTM_param(input_dim, num_mem_cells, learning_rate)
        # input sequence
        self.input_xs = []
        # Node list
        self.lstm_node_list = []

    def add_lstm(self, input_x):
        '''Storing input sequence, add new state everytime there is a new input'''

        self.input_xs.append(input_x)
        lstm_state = LSTM_state(self.lstm_param.num_mem_cells, self.lstm_param.input_dim)
        self.lstm_node_list.append(LSTM_Node(self.lstm_param, lstm_state))

        idx = len(self.input_xs) - 1

        if idx is 0:
            self.lstm_node_list[idx].forward_pass(input_x)
        else:
            s_prev = self.lstm_node_list[idx - 1].state.s
            h_prev = self.lstm_node_list[idx - 1].state.h
            self.lstm_node_list[idx].forward_pass(input_x, s_prev, h_prev)


if __name__ == '__main__':
    pass
