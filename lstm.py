# ----------------------------------------------------------------------
# This is the implemenation of single layer LSTM
# ----------
# Equations:
# ----------
#   g(t) = tanh(Wgx * x(t), Wgh * h(t - 1) + bg)
#   i(t) = tanh(Wix * x(t), Wih * h(t - 1) + bg) ---> input Gate
#   f(t) = tanh(Wfx * x(t), Wfh * h(t - 1) + bg) ---> forget Gate
#   o(t) = tanh(Wox * x(t), Woh * h(t - 1) + bg) ---> output Gate
#   S(t) = g(t) .* i(t) + f(t) .* S(t - 1)  ---> update Cell State s(t)
#   h(t) = S(t) .* o(t) ---> update hidden state
#
# ---------
# Notation:
# ---------
#    * : dot product
#   .* : hadamard product
#
# In implementation we concatenate the weights as below
# Wg = [Wgx Wgh]
# Wi = [Wix Wih]
# Wf = [Wfx Wfh]
# Wo = [Wox Woh]
# Xs(t) = [x(t) h(t-1)]
# ----------------------------------------------------------------------
import numpy as np
import pdb

initial_seed = 42


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


class loss_layer:
    '''
    Computes squared loss
    '''

    def loss(self, pred, label):
        return (pred[0] - label) ** 2

    def loss_grad(self, pred, label):
        dL = np.zeros_like(pred)
        dL[0] = 2 * (pred[0] - label)
        return dL


class LSTM_state:
    ''' Class holding all the states of LSTM unit '''

    def __init__(self, input_dim, output_dim, num_mem_cells):
        '''At time t=0 initialize the states to zeros'''

        self.g = np.zeros(num_mem_cells)
        self.i = np.zeros(num_mem_cells)
        self.f = np.zeros(num_mem_cells)
        self.o = np.zeros(num_mem_cells)
        self.s = np.zeros(num_mem_cells)
        self.h = np.zeros(num_mem_cells)
        self.y = np.zeros(output_dim)
        self.prob = np.zeros(output_dim)


class LSTM_data:

    def __init__(self, input_txt_file):
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
        self.seq_len = 25
        ptr = 0
        self.inputs = [self.char_to_index[ch] for ch in self.data[ptr: ptr + self.seq_len]]
        self.targets = [self.char_to_index[ch] for ch in self.data[ptr + 1: ptr + 1 + self.seq_len]]


class LSTM_param:
    ''' Class holding all the LSTM learnable weights and biases of LSTM unit'''

    def random_array(self, mu, sigma, *shape_args):
        np.random.seed(initial_seed)
        return np.random.rand(*shape_args) * sigma + mu

    def init_param(self, num_mem_cells, output_dim, concat_dim, mu=-0.1, sigma=0.2):
        '''initialize the weights'''

        # Weight initialization
        self.wg = self.random_array(mu, sigma, num_mem_cells, concat_dim)
        self.wi = self.random_array(mu, sigma, num_mem_cells, concat_dim)
        self.wf = self.random_array(mu, sigma, num_mem_cells, concat_dim)
        self.wo = self.random_array(mu, sigma, num_mem_cells, concat_dim)
        self.wy = self.random_array(mu, sigma, output_dim, num_mem_cells)

        # Bias initialization
        self.bg = self.random_array(mu, sigma, num_mem_cells)
        self.bi = self.random_array(mu, sigma, num_mem_cells)
        self.bf = self.random_array(mu, sigma, num_mem_cells)
        self.bo = self.random_array(mu, sigma, num_mem_cells)
        self.by = self.random_array(mu, sigma, output_dim)

        # weight gradient initialization
        self.dwg = np.zeros_like(self.wg)
        self.dwi = np.zeros_like(self.wi)
        self.dwf = np.zeros_like(self.wf)
        self.dwo = np.zeros_like(self.wo)
        self.dwy = np.zeros_like(self.wy)

        # bias gradient initialization
        self.dbg = np.zeros_like(self.bg)
        self.dbi = np.zeros_like(self.bi)
        self.dbf = np.zeros_like(self.bf)
        self.dbo = np.zeros_like(self.bo)
        self.dby = np.zeros_like(self.by)

    def __init__(self, input_dim, output_dim, num_mem_cells=100, lr=0.1):
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
        self.output_dim = output_dim
        self.concat_dim = input_dim + num_mem_cells

        # Init parameters
        self.init_param(num_mem_cells, output_dim, self.concat_dim)


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
        self.y = None

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
        self.state.y = np.dot(self.param.wy, self.state.h) + self.param.by
        pred = self.state.y
        self.state.prob = np.exp(pred) / np.sum(np.exp(pred))

        # store the inputs
        self.x = x
        self.xc = xc
        self.y = self.state.y


class LSTM_network:

    def __init__(self, input_dim, output_dim, num_mem_cells=100, learning_rate=0.1):
        '''Initialize the LSTM unit, LSTM state'''

        # weights and bias are reused, so initialize lstm_param only once
        self.lstm_param = LSTM_param(input_dim, output_dim, num_mem_cells, learning_rate)
        # input sequence
        self.input_xs = []
        # Node list
        self.lstm_node_list = []
        self.loss = 0

    def feed_backward(self, target):
        ''' Backpropogation '''

        assert len(self.lstm_node_list) == len(target)

        dh_next = np.zeros_like(self.lstm_node_list[0].state.h)
        ds_next = np.zeros_like(self.lstm_node_list[0].state.s)

        for i, tt in reversed(list(enumerate(target))):
            param = self.lstm_node_list[i].param
            state = self.lstm_node_list[i].state
            x_dim = param.input_dim
            xc = self.lstm_node_list[i].xc

            dy = np.copy(self.lstm_node_list[i].state.prob)
            dy[tt] -= 1

            # gradient till cell state at time t
            dh = np.dot(param.wy.T, dy) + dh_next
            ds = dh * state.o + ds_next

            # gradients till the non linearities for gates
            dg = state.i * ds
            di = state.g * ds
            df = self.lstm_node_list[i].s_prev * ds
            do = dh * state.s

            # gradients including non-linearities
            dg_input = (1.0 - state.g ** 2) * dg
            di_input = (1.0 - state.i) * state.i * di
            df_input = (1.0 - state.f) * state.f * df
            do_input = (1.0 - state.o) * state.o * do

            # Update gradients
            self.lstm_node_list[i].param.dwy += np.outer(dy, state.h)
            self.lstm_node_list[i].param.dby += dy
            self.lstm_node_list[i].param.dwg += np.outer(dg_input, xc)
            self.lstm_node_list[i].param.dwi += np.outer(di_input, xc)
            self.lstm_node_list[i].param.dwf += np.outer(df_input, xc)
            self.lstm_node_list[i].param.dwo += np.outer(do_input, xc)
            self.lstm_node_list[i].param.dbg += dg_input
            self.lstm_node_list[i].param.dbi += di_input
            self.lstm_node_list[i].param.dbf += df_input
            self.lstm_node_list[i].param.dbo += do_input

            ds_next = state.f * ds
            # compute bottom diff
            dxc = np.zeros_like(xc)
            dxc += np.dot(param.wi.T, di_input)
            dxc += np.dot(param.wf.T, df_input)
            dxc += np.dot(param.wo.T, do_input)
            dxc += np.dot(param.wg.T, dg_input)
            dh_next = dxc[x_dim:]

    def calculate_loss(self, target):
        ''' Cross entropy loss'''

        assert len(self.lstm_node_list) == len(target)

        loss = 0
        for i, tt in enumerate(target):
            prob = self.lstm_node_list[i].state.prob
            loss += -np.log(prob[tt])

        self.loss = loss

    def feed_forward(self, input_x):
        '''Storing input sequence, add new state everytime there is a new input
        @params:
        --------
            input_x - input vector to LSTM, x(t)
        '''

        self.input_xs.append(input_x)
        lstm_state = LSTM_state(self.lstm_param.input_dim, self.lstm_param.output_dim, self.lstm_param.num_mem_cells)
        self.lstm_node_list.append(LSTM_Node(self.lstm_param, lstm_state))

        idx = len(self.input_xs) - 1

        if idx is 0:
            self.lstm_node_list[idx].forward_pass(input_x)
        else:
            s_prev = self.lstm_node_list[idx - 1].state.s
            h_prev = self.lstm_node_list[idx - 1].state.h
            self.lstm_node_list[idx].forward_pass(input_x, s_prev, h_prev)


if __name__ == '__main__':
    np.random.seed(initial_seed)

    objLstmData = LSTM_data('./input.txt')
    input_dim, output_dim, num_iters, num_samples = objLstmData.vocab_size, objLstmData.vocab_size, 100, objLstmData.seq_len
    objLstmNet = LSTM_network(input_dim, output_dim)

    x_list = objLstmData.inputs
    y_list = objLstmData.targets

    for i in range(num_samples):
        x_one_hot = np.zeros((output_dim))
        x_one_hot[x_list[i]] = 1
        objLstmNet.feed_forward(x_one_hot)

    objLstmNet.calculate_loss(y_list)
    objLstmNet.feed_backward(y_list)
