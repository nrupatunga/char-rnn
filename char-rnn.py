#  !/usr/bin
#  Toy example of seq2seq modeling using vanilla RNN
#  h(t) = tanh(Wxh*x(t) + Whh*h(t-1) + bxh + bhh)
#  y(t) = Wyh*h(t) + byh
#  x(t) is n dimensional embedding for one hot vector for vocabulary of length n
import numpy as np
from random import uniform


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

        data = []
        with open(txtFile, 'r') as f:
            data = f.read()

        self.data = data
        self.data_len = len(data)
        self.lr = lr

        chars = list(set(data))  # list of unique characters
        vocab_size = len(chars)

        #  Characters to index and vice versa
        self.char_to_index = {c: i for i, c in enumerate(chars)}
        self.index_to_char = {i: c for i, c in enumerate(chars)}
        self.vocab_size = vocab_size
        self.num_hidden_params = num_hidden_units

        np.random.seed(42)
        #  Initialize weights of RNN, with normal distribution with sigma = 0.01
        self.wxh = np.random.randn(num_hidden_units, vocab_size) * 0.01
        self.whh = np.random.randn(num_hidden_units, num_hidden_units) * 0.01
        self.why = np.random.randn(self.vocab_size, num_hidden_units) * 0.01

        #  Initialize bias
        self.bh = np.zeros((num_hidden_units, 1))
        self.by = np.zeros((vocab_size, 1))

    def sample(self, h, seed_ix, n):
        ''' Sample the sequence of outputs'''

        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        for t in range(n):
            h = np.tanh(np.dot(self.wxh, x) + np.dot(self.whh, h) + self.bh)
            y = np.dot(self.why, h) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)

        return ixes

    def forward_backward(self, X, T, h_prev):
        '''forward pass'''

        x_one_hot, h, y_pred, prob, loss = {}, {}, {}, {}, 0
        h[-1] = np.copy(h_prev)
        for t in range(len(X)):
            x_one_hot[t] = np.zeros((self.vocab_size, 1))
            x_one_hot[t][X[t]] = 1

            #  RNN forward pass equations
            h[t] = np.tanh(np.dot(self.wxh, x_one_hot[t]) + np.dot(self.whh, h[t - 1]) + self.bh)
            y_pred[t] = np.dot(self.why, h[t]) + self.by
            prob[t] = np.exp(y_pred[t]) / np.sum(np.exp(y_pred[t]))
            loss += -np.log(prob[t][T[t], 0])

        #  backward pass
        dwxh, dwhh = np.zeros_like(self.wxh), np.zeros_like(self.whh)
        dwhy, dbh, dby = np.zeros_like(self.why), np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(h[0])
        for t in reversed(range(len(X))):
            dy = np.copy(prob[t])
            dy[T[t]] -= 1
            dwhy += np.dot(dy, h[t].T)
            dby += dy

            dh = np.dot(self.why.T, dy) + dhnext
            dhraw = (1 - h[t] * h[t]) * dh
            dbh += dhraw
            dwxh += np.dot(dhraw, x_one_hot[t].T)
            dwhh += np.dot(dhraw, h[t - 1].T)
            dhnext = np.dot(self.whh.T, dhraw)

        for dparam in [dwxh, dwhh, dwhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        return loss, dwxh, dwhh, dwhy, dbh, dby, h[len(X) - 1]

    #  gradient checking
    def gradCheck(self, inputs, targets, hprev):
        num_checks, delta = 10, 1e-5
        _, dWxh, dWhh, dWhy, dbh, dby, _ = self.forward_backward(inputs, targets, hprev)
        for param, dparam, name in zip([self.wxh, self.whh, self.why, self.bh, self.by], [dWxh, dWhh, dWhy, dbh, dby], ['Wxh', 'Whh', 'Why', 'bh', 'by']):
            s0 = dparam.shape
            s1 = param.shape
            assert s0 == s1
            print name
            for i in xrange(num_checks):
                ri = int(uniform(0, param.size))
                # evaluate cost at [x + delta] and [x - delta]
                old_val = param.flat[ri]
                param.flat[ri] = old_val + delta
                cg0, _, _, _, _, _, _ = self.forward_backward(inputs, targets, hprev)
                param.flat[ri] = old_val - delta
                cg1, _, _, _, _, _, _ = self.forward_backward(inputs, targets, hprev)
                param.flat[ri] = old_val  # reset old value for this parameter
                # fetch both numerical and analytic gradient
                grad_analytic = dparam.flat[ri]
                grad_numerical = (cg0 - cg1) / (2 * delta)
                rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
                print '%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error)

    def train(self):
        ''' Training the RNN'''

        ptr, sample_n = 0, 0
        h_prev = np.zeros((self.num_hidden_params, 1))
        mwxh, mwhh, mwhy = np.zeros_like(self.wxh), np.zeros_like(self.whh), np.zeros_like(self.why)
        mbh, mby = np.zeros_like(self.bh), np.zeros_like(self.by)  # memory variables for Adagrad
        smooth_loss = -np.log(1.0 / self.vocab_size) * self.seq_len  # loss at iteration 0

        while True:
            if ptr + self.seq_len >= self.data_len or sample_n is 0:
                h_prev = np.zeros((self.num_hidden_params, 1))
                ptr = 0

            inputs = [self.char_to_index[ch] for ch in self.data[ptr: ptr + self.seq_len]]
            targets = [self.char_to_index[ch] for ch in self.data[ptr + 1: ptr + 1 + self.seq_len]]
            # self.gradCheck(inputs, targets, h_prev)
            if sample_n % 100 == 0:
                sample_ix = self.sample(h_prev, inputs[0], 200)
                txt = ''.join(self.index_to_char[ix] for ix in sample_ix)
                print('\n {} \n'.format(txt))

            loss, dwxh, dwhh, dwhy, dbh, dby, hprev = self.forward_backward(inputs, targets, h_prev)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            if sample_n % 100 is 0:
                print('Iteration {}, loss = {}'.format(sample_n, smooth_loss))

            mwxh += dwxh * dwxh
            self.wxh += -self.lr * dwxh / np.sqrt(mwxh + 1e-8)

            mwhh += dwhh * dwhh
            self.whh += -self.lr * dwhh / np.sqrt(mwhh + 1e-8)

            mwhy += dwhy * dwhy
            self.why += -self.lr * dwhy / np.sqrt(mwhy + 1e-8)

            mbh += dbh * dbh
            self.bh += -self.lr * dbh / np.sqrt(mbh + 1e-8)

            mby += dby * dby
            self.by += -self.lr * dby / np.sqrt(mby + 1e-8)

            ptr = ptr + self.seq_len
            sample_n = sample_n + 1


if __name__ == '__main__':
    objRNN = VanillaRNN('input.txt')
    objRNN.train()
