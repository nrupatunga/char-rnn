#  !/usr/bin
#  PEP8
#  Toy example of seq2seq modeling using vanilla RNN
#  h(t) = tanh(Wxh*x(t) + Whh*h(t-1) + bxh + bhh)
#  y(t) = Wyh*h(t) + byh
#  x(t) is n dimensional embedding for one hot vector for vocabulary of length n

import numpy as np
import pdb


class VanillaRNN(object):
    '''functions to train basic rnn'''

    seq_len = 25

    def __init__(self, txtFile, num_hidden_units=100, seq_len=25, lr=1e-1):
        ''' Initialize variables,  character indexing
        @params:
        --------
            txtFile(optional) - text file input containing the test characters
            num_hidden_units  - Number of hidden units
        '''

        pdb.set_trace()
        data = []
        with open(txtFile, 'r') as f:
            data = f.read()

        self.data_len = len(data)
        chars = list(set(data))  # list of unique characters
        vocab_size = len(chars)

        #  Characters to index and vice versa
        self.char_to_index = {i: c for i, c in enumerate(chars)}
        self.index_to_char = {c: i for i, c in enumerate(chars)}
        self.vocab_size = vocab_size
        self.num_hidden_params = num_hidden_units

        #  Initialize weights of RNN, with normal distribution with sigma = 0.01
        self.wxh = np.random.randn(num_hidden_units, vocab_size) * 0.01
        self.whh = np.random.randn(num_hidden_units, num_hidden_units) * 0.01
        self.why = np.random.randn(self.vocab_size, num_hidden_units) * 0.01

        #  Initialize bias
        self.bh = np.random.randn(num_hidden_units, 1)
        self.by = np.random.randn(vocab_size, 1)

    def forward_backward(self, X, T, h_prev):
        '''forward pass'''

        x_one_hot, h, y_pred, prob, loss = [], [], [], 0
        h[-1] = h_prev
        for t in range(len(X)):
            x_one_hot = np.zeros(self.vocab_size, 1)
            x_one_hot[t][X[t]] = 1

            #  RNN forward pass equations
            h[t] = np.tanh(np.dot(self.wxh, x_one_hot) + np.dot(self.whh, h[t - 1]) + self.bh)
            y_pred[t] = np.dot(self.why, h[t]) + self.by
            prob[t] = np.exp(y_pred[t]) / np.sum(y_pred[t])
            loss += -np.log(prob[t][T[t]])

    def train(self):
        ''' Training the RNN'''

        ptr, sample_n = 0, 0
        h_prev = np.zeros(self.num_hidden_params, 1)

        if ptr + self.seq_len >= self.data_len or sample_n is 0:
            h_prev = np.zeros(self.num_hidden_params, 1)
            ptr = 0

        inputs = [self.char_to_index[ch] for ch in self.data[ptr: ptr + self.seq_len]]
        targets = [self.index_to_char[ch] for ch in self.data[ptr + 1: ptr + 1 + self.seq_len]]
        self.forward_backward(inputs, targets, h_prev)


if __name__ == '__main__':
    objRNN = VanillaRNN('input.txt')
