import numpy as np
from scipy.signal import convolve2d


class Layer_Conv2D:
    def __init__(self, kernel_size, padding='valid'):
        self.k_h, self.k_w = kernel_size
        self.weights = 0.01 * np.random.randn(self.k_h, self.k_w)
        self.padding = padding
        self.biases = np.zeros(1)
        
    def forward(self, input):
        if self.padding == 'valid':
            self.input = input
            self.output = np.zeros((input.shape[0] - self.k_h + 1, input.shape[1] - self.k_w + 1))
            
        elif self.padding == 'same':
            if self.k_h % 2 == 1 and self.k_h % 2 == 1:
                self.p_h = (self.k_h - 1) // 2
                self.p_w = (self.k_w - 1) // 2
                self.p_h_top, self.p_h_bottom = self.p_h, self.p_h
                self.p_w_left, self.p_w_right = self.p_w, self.p_w
            else:
                self.p_h_top = (self.k_h - 1) // 2
                self.p_h_bottom = self.k_h // 2
                self.p_w_left = (self.k_w - 1) // 2
                self.p_w_right = self.k_w // 2
                
        
            self.input = np.pad(input, pad_width=((self.p_h_top, self.p_h_bottom), 
                                          (self.p_w_left, self.p_w_right)), mode='constant')
            
            self.output = np.zeros((input.shape[0], input.shape[0]))
        else:
            raise ValueError('Padding must be "valid" or "same".')
        for row in range(self.output.shape[0]):
            for column in range(self.output.shape[1]):
                sub_array = self.input[row:row+self.k_h, column:column+self.k_w]
                self.output[row, column] = np.sum(sub_array*self.weights) + self.biases
            
    def backward(self, dvalues):
        self.dweights = np.zeros_like(self.weights)
        self.dinputs = np.zeros_like(self.input, dtype=float)
        for i in range(dvalues.shape[0]):
            for j in range(dvalues.shape[1]):
                self.dweights+= self.input[i:i+self.k_h, j:j+self.k_w] * dvalues[i, j]
                self.dinputs[i:i+self.k_h, j:j+self.k_w] += self.weights * dvalues[i, j]
        if self.padding == 'same':
            self.dinputs = self.dinputs[self.p_h_top:-self.p_h_bottom, self.p_w_left:-self.p_w_right]
        self.dbiases = np.sum(dvalues, axis=(0, 1))