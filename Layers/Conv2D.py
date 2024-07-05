import numpy as np
from scipy.signal import convolve2d


class Layer_Conv2D:
    def __init__(self, kernel_size, padding='valid'):
        self.k_h, self.k_w = kernel_size
        self.weights = 0.01 * np.random.randn(self.k_h, self.k_w)
        self.padding = padding
        self.biases = np.zeros(1)
        
    def forward(self, input):
        self.input = input
        self.output = convolve2d(self.input, np.flipud(np.fliplr(self.weights)), mode=self.padding, fillvalue=0) + self.biases
        
        if self.padding == 'same':
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
    def backward(self, dvalues):
        self.dweights = convolve2d(self.input, np.flipud(np.fliplr(dvalues)), mode='valid')
        self.dinputs = convolve2d(dvalues, np.flipud(np.fliplr(self.weights)),mode=self.padding)
    
        self.dbiases = np.sum(dvalues, axis=(0, 1))
